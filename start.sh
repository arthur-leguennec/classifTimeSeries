### Test script

# model = leNet2
for fichier in ../UCR_TS_Archive_2015/*
do
    echo $(basename $fichier .${fichier##*.})
    echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.001 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.002 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.005 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.01  -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.02  -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.05  -lrd 0.0005 -iter 300
done


# model = leNet3
for fichier in ../UCR_TS_Archive_2015/*
do
    echo $(basename $fichier .${fichier##*.})
    echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet3 -lr 0.001 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet3 -lr 0.002 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet3 -lr 0.005 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet3 -lr 0.01  -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet3 -lr 0.02  -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet3 -lr 0.05  -lrd 0.0005 -iter 300
done


# params : lr = 0.005, iter = 100, script = true, model = leNet1
for fichier in ../UCR_TS_Archive_2015/*
do
    echo $(basename $fichier .${fichier##*.})
    echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet1 -lr 0.001 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet1 -lr 0.002 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet1 -lr 0.005 -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet1 -lr 0.01  -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet1 -lr 0.02  -lrd 0.0005 -iter 300
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet1 -lr 0.05  -lrd 0.0005 -iter 300
done



# model = mcdcnn
for fichier in ../UCR_TS_Archive_2015/*
do
    echo $(basename $fichier .${fichier##*.})
    echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.001 -lrd 0.0005 -iter 200
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.002 -lrd 0.0005 -iter 200
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.005 -lrd 0.0005 -iter 200
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.01  -lrd 0.0005 -iter 200
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.02  -lrd 0.0005 -iter 200
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.05  -lrd 0.0005 -iter 200
done
