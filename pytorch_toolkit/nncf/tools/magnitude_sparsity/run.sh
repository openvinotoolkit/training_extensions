#!/usr/bin/env bash

print_usage() {
cat << EOF

usage: $0 -d PATH_TO_DATA -m,--model PATH_TO_IR_XML [-b,--bin PATH_TO_BIN_XML]
       [-o,--output OUTPUT_DIR] [-h,--help]
EOF
}

OPTIONS=hm:b:a:o:d:
LONGOPTIONS=help,model:,bin:,annotation:,output:,data:

getopt --test > /dev/null
if [[ $? -ne 4 ]]; then
    exit 1
fi

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    exit 2
fi
eval set -- "$PARSED"

while true; do
    case "$1" in
        -m|--model)
            xml_file="$2"
            shift 2
            ;;
        -b|--bin)
            bin_file="$2"
            shift 2
            ;;
        -o|--output)
            output="$2"
            shift 2
            ;;
        -d|--data)
            data_dir="$2"
            shift 2
            ;;
        -a|--annotation)
            annotation_file="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Option error"
            exit 3
            ;;
    esac
done

if [[ -z ${annotation_file} ]]; then
    echo "Annotation file is required for acc check (-a/--annotation option)"
    print_usage
    exit 1
fi

if [[ -z ${data_dir} ]]; then
    echo "Path to dataset is required for acc check (-d/--data option)"
    print_usage
    exit 1
fi

if [[ -z ${xml_file} ]]; then
    echo "XML file is required (-m/--model option)"
    print_usage
    exit 1
fi

if [[ -z ${bin_file} ]]; then
    bin_file=${xml_file/.xml/.bin}
fi

if [[ -z ${output} ]]; then
    output=models
fi

echo "Input XML: ${xml_file}"
echo "Input BIN: ${bin_file}"
echo "Annotation file: ${annotation_file}"
echo "Output dir: ${output}"
echo "Data dir: ${data_dir}"

mkdir -p $output


for i in {0..90..5}
do  
    new_bin="${output}/$(basename $bin_file)"
    new_bin=${new_bin/.bin/_sp$i.bin}

    echo
    echo "Trying sparsity level $i%, new bin file: ${new_bin}"
    python3 sparsify.py -m $xml_file -b $bin_file -o $new_bin -s $i

    cp accuracy_check_config_template.yaml _config.yaml

    sed -i -e "s@<XML_FILE>@$xml_file@g" _config.yaml
    sed -i -e "s@<BIN_FILE>@$new_bin@g" _config.yaml
    sed -i -e "s@<ANNOTATIONS_FILE>@$annotation_file@g" _config.yaml
    sed -i -e "s@<DATA_DIR>@$data_dir@g" _config.yaml

    accuracy_check -m . -s . -a . -c _config.yaml
    rm _config.yaml

done
