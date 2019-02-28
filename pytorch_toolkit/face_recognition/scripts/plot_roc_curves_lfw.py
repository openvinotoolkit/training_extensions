"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse

import matplotlib.pyplot as plt
from evaluate_lfw import get_auc

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('rocs', metavar='ROCs', type=str, nargs='+',
                        help='paths to roc curves')

    args = parser.parse_args()

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()

    for curve_file in args.rocs:
        fprs = []
        tprs = []
        with open(curve_file, 'r') as f:
            for line in f.readlines():
                values = line.strip().split()
                fprs.append(float(values[1]))
                tprs.append(float(values[0]))

        curve_name = curve_file.split('/')[-1].split('.')[0]
        plt.plot(fprs, tprs, label=curve_name)
        plt.legend(loc='best', fontsize=10)

        print('AUC for {}: {}'.format(curve_name, get_auc(fprs, tprs)))

    plt.show()

if __name__ == '__main__':
    main()
