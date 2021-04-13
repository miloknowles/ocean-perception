#pragma once

#include <glog/logging.h>

#include "dataset/data_provider.hpp"
#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "dataset/caddy_dataset.hpp"
#include "dataset/acfr_dataset.hpp"

#include "core/path_util.hpp"

namespace bm {
namespace dataset {


// Integer representation for all of the datasets that we support.
// Allows changing the dataset type via a YAML integer param.
enum Dataset
{
  FARMSIM = 0,
  CADDY = 1,
  HIMB = 2,
  ACFR = 3,
  ZEDM = 4
};


// Convenience function for returning a dataset based on the enum type specified.
// Pass in the top level dataset folder, and optionally a subfolder if required (e.g HIMB "train").
// Returns the dataset and sets shared_params_path to the relevant dataset params in
// vehicle/config/auv_base/*.
inline DataProvider GetDatasetByName(Dataset code,
                                     const std::string& folder,
                                     const std::string& subfolder,
                                     std::string& shared_params_path)
{
  dataset::DataProvider dataset;

  switch (code) {
    case Dataset::FARMSIM:
      dataset = dataset::EurocDataset(folder);
      shared_params_path = config_path("auv_base/Farmsim.yaml");
      break;
    case Dataset::CADDY:
      throw std::runtime_error("No config available for CADDY");
      dataset = dataset::CaddyDataset(folder, subfolder);
      break;
    case Dataset::HIMB:
      dataset = dataset::HimbDataset(folder, subfolder);
      shared_params_path = config_path("auv_base/HIMB.yaml");
      break;
    case Dataset::ACFR:
      dataset = dataset::AcfrDataset(folder);
      shared_params_path = config_path("auv_base/ACFR.yaml");
      break;
    case Dataset::ZEDM:
      dataset = dataset::EurocDataset(folder);
      shared_params_path = config_path("auv_base/ZEDMini.yaml");
      break;
    default:
      LOG(FATAL) << "Unknown dataset type: " << code << std::endl;
      break;
  }

  return dataset;
}

}
}
