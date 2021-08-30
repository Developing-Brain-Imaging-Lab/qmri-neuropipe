#!/bin/bash

chmod +x shared/mcdespot_condor

in_dir=$1
out_dir=$2
chunks=$3

echo ./shared/mcdespot_condor --spgr=${in_dir}/multiflipSPGR_measurements.mcd --ssfp=${in_dir}/multiflipSSFP_measurements.mcd --params=shared/mcd_params.json --b1=${in_dir}/b1_measurements.mcd --f0=${in_dir}/f0_measurements.mcd --out_dir=${out_dir} --algo=GRC --chunk_size=${chunks} -v

./shared/mcdespot_condor --spgr=${in_dir}/multiflipSPGR_measurements.mcd --ssfp=${in_dir}/multiflipSSFP_measurements.mcd --params=shared/mcd_params.json --b1=${in_dir}/b1_measurements.mcd --f0=${in_dir}/f0_measurements.mcd --out_dir=${out_dir} --algo=GRC --chunk_size=${chunks} -v
