#pragma once
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

namespace qmri::io {

template<typename T=float, unsigned Dim=4>
using Image = itk::Image<T, Dim>;
template<typename T=float, unsigned Dim=4>
using Reader = itk::ImageFileReader<Image<T,Dim>>;

template<typename T=float, unsigned Dim>
inline Image<T,Dim>::Pointer read_image(const std::string& path){
    auto r = Reader<T,Dim>::New();
    r->SetFileName(path);
    r->Update();
    
    return r->GetOutput();
    
}

template<typename Img4D>
inline void pack_soa(const Img4D* img, Eigen::MatrixXf& X) {
  auto region = img->GetLargestPossibleRegion();
  auto size = region.GetSize();
  int NX=size[0], NY=size[1], NZ=size[2], NT=size[3];
  int V = NX*NY*NZ;
  X.resize(NT, V);
  for (int z=0; z<NZ; ++z) for (int y=0; y<NY; ++y) for (int x=0; x<NX; ++x){
    int v=(z*NY + y)*NX + x;
    for (int t=0; t<NT; ++t){
      typename Img4D::IndexType idx{{x,y,z,t}};
      X(t,v) = img->GetPixel(idx);
    }
  }
}

template<typename Img3D>
inline void write_map(const Eigen::VectorXf& vec, int NX,int NY,int NZ, const std::string& fn){
  auto out = Img3D::New();
  typename Img3D::RegionType r;
  typename Img3D::IndexType idx{{0,0,0}};
  typename Img3D::SizeType sz{{(unsigned long)NX,(unsigned long)NY,(unsigned long)NZ}};
  r.SetIndex(idx); r.SetSize(sz);
  out->SetRegions(r); out->Allocate();
  for (int z=0; z<NZ; ++z) for (int y=0; y<NY; ++y) for (int x=0; x<NX; ++x){
    int v=(z*NY + y)*NX + x;
    typename Img3D::IndexType j{{x,y,z}};
    out->SetPixel(j, vec[v]);
  }
  using Writer=itk::ImageFileWriter<Img3D>;
  auto w=Writer::New(); w->SetFileName(fn); w->SetInput(out); w->Update();
}
} // namespace qmri::io
