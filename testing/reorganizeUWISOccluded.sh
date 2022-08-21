find . -mindepth 2 -type f -print -exec mv --backup=numbered {} . \;
rm -r Lounge
rm -r Warehouse
unzip '*.zip'
rm *.zip
