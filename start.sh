### Test script

# params : lr = 0.005, iter = 100, script = true, model = leNet1
for fichier in ../UCR_TS_Archive_2015/*
do
    echo $(basename $fichier .${fichier##*.})
    echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet1 -lr 0.005 -iter 100
done
