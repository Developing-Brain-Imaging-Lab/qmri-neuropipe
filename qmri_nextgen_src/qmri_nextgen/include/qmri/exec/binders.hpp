#pragma once
#include <Eigen/Dense>
#include <type_traits>

namespace qmri {

// Identity binder: returns the same model for every voxel.
// Useful when a model has no voxel-specific constants to bind.
template <typename Model>
struct IdentityBinder {
    const Model& model;
    inline Model operator()(int /*v*/) const { return model; } // returns a small copy
};

// If you prefer avoiding copies and 'Model' is cheap to copy anyway, this is fine.
// If your model is heavy, you can change to return a lightweight wrapper
// that carries references, or return by const ref and adjust callsites.

} // namespace qmri
