#!/bin/bash
# Usage: ./download.sh [ip] [port]
set -e

# useful variables
host=$1
port=$2
data_dir=data_$(echo "$host" | cut -d':' -f 1)

# create output directory
if [ -d "$data_dir" ]; then
    rm -r $data_dir
fi
mkdir $data_dir

# get list of measurements
measurements=$(curl -s -G http://"$host":"$port"/query --data-urlencode "q=SHOW measurements ON monitoring" -H "Accept: application/csv" | cut -d',' -f3 | tail -n +2)

# download and store in separate csv files
for var in $measurements;
do
    echo -e Downloading "\033[1m"$var"\033[0m" ...
    docker run --rm influxdb influx -host $host -port $port -database 'monitoring' -execute 'SELECT * from '"$var" -format csv > $data_dir/$var.csv
done

echo -e "\nSummary:"
du $data_dir/* -ch
echo

poetry run python utils/merge_tables.py $data_dir
