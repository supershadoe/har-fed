set -e
if [ -d PAMAP2_Dataset ]; then
    exit
fi
if [ ! -f dataset.zip ]; then
    curl -L -o dataset.zip https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip
fi
unzip dataset.zip
rm readme.pdf
unzip PAMAP2_Dataset.zip
rm dataset.zip
cd ..
