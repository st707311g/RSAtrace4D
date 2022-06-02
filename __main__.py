import argparse
import json
import os
import sys
from glob import glob

from numpy import source

from __common import DESCRIPTION, logger
from __modules import (DataMaker, Fitting, ModelPredicting, ModelTraining,
                       Quantification, RSAtrace4D_Series)
from __rinfo import RSA_Vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION) 
    parser.add_argument('-s', '--source', type=str, help='Indicate source directory')

    args = parser.parse_args()

    if args.source is None:
        parser.print_help()
        sys.exit()

    source_directory: str = args.source
    if source_directory.endswith('/') or source_directory.endswith('\\'):
        source_directory = source_directory[:-1]
    
    if not os.path.isdir(source_directory):
        logger.error(f'Indicate valid virectory path.')
        exit()

    root_directory_list = sorted(glob(source_directory+'/**/.*/', recursive=True))
    root_directory_list = [root_directory for root_directory in root_directory_list if RSAtrace4D_Series(root_directory=root_directory).is_valid()]

    for root_directory in root_directory_list:
        volume_series = RSAtrace4D_Series(root_directory=root_directory)
        rinfo_file_at_last_day = rinfo_file=volume_series.rinfo_file_list[0]
        result_csv_rsa_params = os.path.join(root_directory, '.RSAtrace4D_RSA_params.csv')

        data_maker = DataMaker(
            source_dir=volume_series.volume_list[-1], 
            rinfo_file=volume_series.rinfo_file_list[0]
        )
        data_maker.load_segments()
        x_list, y_list = data_maker.get_all_train_data_set()

        model_training = ModelTraining(
            model_directory=volume_series.model_destination
        )
        model_training.core_model_training(
            data_maker=data_maker
        )
        model_training.individual_model_training(
            data_maker=data_maker
        )

        result_out_path = os.path.join(root_directory, '.RSAtrace4D_raw_result.csv')
        if os.path.isfile(result_out_path):
            logger.info(f'[skip] RSAtrace4D result file already exists: {result_out_path}')
        else:
            model_predicting = ModelPredicting(
                dir_list=volume_series.volume_list,
                model_dir_path=volume_series.model_destination,
                rsa_vector=data_maker.rsa_vector
            )
            df = model_predicting.proceed_all()
            with open(result_out_path, 'w', newline="") as f:
                df.to_csv(f, header=False)

        fitting = Fitting(
            source_csv=result_out_path
        )

        fitting.save_plot_all(fitting=False)
        fitting.save_plot_all(fitting=True)

        rsa_vector_last = RSA_Vector()
        with open(rinfo_file_at_last_day, 'r') as f:
            trace_dict = json.load(f)

        rsa_vector_last.load_from_dict(trace_dict=trace_dict, file=rinfo_file_at_last_day)

        quantification = Quantification(
            rsa_vector=rsa_vector_last,
            fitting=fitting
        )

        df = quantification.get_df()
        with open(result_csv_rsa_params, 'w', newline="") as f:
            df.to_csv(f, header=True)
        