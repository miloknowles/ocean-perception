#pragma once

namespace bm {
namespace core {


struct PinholeModel final {
  PinholeModel(float _fx, float _fy, float _cx, float _cy, float _h, float _w)
      : fx(_fx), fy(_fy), cx(_cx), cy(_cy), height(_h), width(_w) {}

  float fx = 0;
  float fy = 0;
  float cx = 0;
  float cy = 0;

  // Nominal width and height for an image captured by this camera.
  int height = 480;
  int width = 752;

  PinholeModel Rescale(int new_height, int new_width) const
  {
    float height_sf = new_height / static_cast<float>(height);
    float width_sf = new_width / static_cast<float>(width);
    return PinholeModel(fx * width_sf, fy * height_sf, cx * width_sf, cy * height_sf,
                        new_height, new_width);
  }
};

}
}
