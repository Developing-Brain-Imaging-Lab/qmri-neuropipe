#pragma once
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
  #include <unistd.h>
  #include <sys/ioctl.h>
#endif

namespace qmri {

class ProgressBar {
public:
  using clock = std::chrono::steady_clock;

  explicit ProgressBar(int total,
                       std::string label = "",
                       int width = 40,
                       int indent = 0,
                       std::chrono::milliseconds min_update = std::chrono::milliseconds(100),
                       bool enabled = true,
                       std::ostream& stream = std::cerr)
  : total_(total),
    label_(std::move(label)),
    width_(width),
    indent_(indent),
    min_update_(min_update),
    enabled_(enabled && is_tty(stream)),                 // disable if not a TTY
    os_(stream.rdbuf())
  {
    last_ = clock::now();
    if (total_ < 0) total_ = 0;
    if (width_ < 1) width_ = 1;
    if (indent_ < 0) indent_ = 0;
  }

  inline void update(int done) {
    if (!enabled_ || total_ <= 0) return;

    // throttle
    auto now = clock::now();
    if (done < total_ && (now - last_) < min_update_) return;

    // compute fraction
    const double frac = clamp01(static_cast<double>(done) / static_cast<double>(total_));

    // compute safe width to avoid wrapping
    const int cols = term_cols_or_default();
    // overhead: space + '[' + ']' + space + "100.0%" (~9 chars) + safety
    const int overhead = indent_ + static_cast<int>(label_.size()) + 1 + 2 + 1 + 7 + 2;
    int bar_width = std::max(1, std::min(width_, std::max(10, cols - overhead)));

    const int filled = static_cast<int>(std::round(frac * bar_width));

    // Build line
    scratch_.clear();
    scratch_.reserve(static_cast<size_t>(indent_ + label_.size() + bar_width + 32));

    // ensure single-line redraw:
    // 1) move cursor to start
    scratch_.push_back('\r');

    // 2) indent + label
    scratch_.append(static_cast<size_t>(indent_), ' ');
    if (!label_.empty()) {
      scratch_.append(label_);
      scratch_.push_back(' ');
    }

    // 3) bar
    scratch_.push_back('[');
    for (int i = 0; i < bar_width; ++i) {
      if (i < filled)       scratch_.push_back('=');
      else if (i == filled) scratch_.push_back('>');
      else                  scratch_.push_back(' ');
    }
    scratch_.push_back(']');

    // 4) percentage
    scratch_.push_back(' ');
    char pct[16];
    std::snprintf(pct, sizeof(pct), "%5.1f%%", frac * 100.0);
    scratch_.append(pct);

    // 5) clear to end of line (erase leftovers if this line shrank)
    scratch_.append("\x1b[K");

    // print atomically
    {
      std::lock_guard<std::mutex> lk(print_mtx());
      os_ << scratch_;
      os_.flush();
    }

    last_ = now;

    if (done >= total_) finish();
  }

  inline void finish() {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lk(print_mtx());
    os_ << '\n';
    os_.flush();
  }

  inline void set_label(std::string label) { label_ = std::move(label); }
  inline void set_indent(int indent) { indent_ = (indent < 0 ? 0 : indent); }
  inline void set_enabled(bool e) { enabled_ = e && is_tty(std::cerr); }

private:
  static inline double clamp01(double x){ return x < 0 ? 0 : (x > 1 ? 1 : x); }

  static bool is_tty(std::ostream& os) {
#if defined(__unix__) || defined(__APPLE__)
    // best-effort: assume cerr → STDERR, cout → STDOUT
    int fd = (&os == &std::cerr) ? STDERR_FILENO : STDOUT_FILENO;
    return isatty(fd);
#else
    (void)os; return true;
#endif
  }

  static int term_cols_or_default() {
#if defined(__unix__) || defined(__APPLE__)
    struct winsize w{};
    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
    return 100;
#else
    return 100;
#endif
  }

  // one mutex for all progress bars to serialize redraws
  static std::mutex& print_mtx() {
    static std::mutex m;
    return m;
  }

  int total_;
  std::string label_;
  int width_;
  int indent_;
  std::chrono::milliseconds min_update_;
  bool enabled_;
  std::ostream os_;
  clock::time_point last_;
  std::string scratch_;
};

} // namespace qmri
