import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from glob import glob
from typing import List

import pandas as pd

from __common import DESCRIPTION, logger
from __modules import RSAtrace4D_Series
from __rinfo import RSA_Vector


@dataclass(frozen=True)
class VectorTrimmer(object):
    rsa_vector_last: RSA_Vector
    volume_directory_list: List[str]
    data_frame: pd.DataFrame

    @property
    def time_point_list(self):
        regex = re.compile('\d+') #type:ignore
        return [int(regex.findall(d)[0]) for d in self.volume_directory_list]

    @property
    def ID_string_list(self):
        ID_string_list = []
        for base_node in self.rsa_vector_last:
            for root_node in base_node:
                ID_string_list.append(root_node.ID_string())
        return ID_string_list

    def get_rsa_vector(self, time_point: int, volume_name: str):
        assert min(self.time_point_list) <= time_point <=  max(self.time_point_list)

        trimmed_rsa_vector = RSA_Vector()
        trimmed_rsa_vector.annotations = self.rsa_vector_last.annotations.copy()
        trimmed_rsa_vector.annotations['volume_name'] = volume_name

        base_node = self.rsa_vector_last.base_node(1)
        trimmed_rsa_vector.append_base(annotations=base_node.annotations.copy())
        trimmed_rsa_vector_base = trimmed_rsa_vector.base_node(1)
        for root_node in base_node:
            root_id = root_node.ID
            annotations = root_node.annotations.copy()
            trimmed = self.data_frame[self.data_frame['root_id']==root_id]
            
            assert len(trimmed) <= 1

            if len(trimmed) == 0 or time_point > trimmed.iat[0, 2]:
                relative_position = 1.
            else:
                if trimmed.iat[0, 1]<= time_point <= trimmed.iat[0, 2]:
                    period = int(trimmed.iat[0, 2] - trimmed.iat[0, 1])
                    relative_position = (time_point - int(trimmed.iat[0, 1]))/period
                    assert 0 <= relative_position <= 1
                else:
                    continue

            if relative_position == 0:
                continue

            annotations['polyline'] = annotations['polyline'][int(len(annotations['polyline'])*(1-relative_position)):]

            trimmed_rsa_vector_base.append(annotations=annotations, rootID=root_id)
            root_node = trimmed_rsa_vector.root_node(rootID=root_id)
            root_node.append(annotations={'coordinate': annotations['polyline'][0]}, interpolation=False)

        return trimmed_rsa_vector

    def get_rsa_vector_all(self):
        for time_point, directory_name in zip(self.time_point_list, self.volume_directory_list):
            predicted_rsa_vector = self.get_rsa_vector(time_point=time_point, volume_name=os.path.basename(directory_name[:-1]))

            destination_dir = os.path.dirname(directory_name[:-1])+'/.predicted_RSA_vectors'
            os.makedirs(destination_dir, exist_ok=True)
            predicted_rsa_vector.save(f'{destination_dir}/{os.path.basename(directory_name[:-1])}.rinfo')

if __name__ == '__main__':
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

        rsa_vector = RSA_Vector()
        with open(rinfo_file_at_last_day, 'r') as f:
            rinfo_dict = json.load(f)

        rsa_vector.load_from_dict(trace_dict=rinfo_dict, file=rinfo_file_at_last_day)
        df = pd.read_csv(result_csv_rsa_params, usecols=['root_id', 'start_of_elongation', 'end_of_elongation'])
        vector_trimmer = VectorTrimmer(
            rsa_vector_last=rsa_vector,
            volume_directory_list=volume_series.volume_list,
            data_frame=df
        )
        
        vector_trimmer.get_rsa_vector_all()
