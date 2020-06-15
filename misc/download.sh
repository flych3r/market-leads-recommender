#!/bin/bash
mkdir -p ../data
wget -i links.txt -P ../data
unzip ../data/estaticos_market.csv.zip -d ../data
rm ../data/estaticos_market.csv.zip
rm -rf ../data/__MACOSX/
zip -9 -j ../data/estaticos_market.zip ../data/estaticos_market.csv
rm ../data/estaticos_market.csv
