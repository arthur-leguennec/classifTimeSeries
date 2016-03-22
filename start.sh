### Test script

# model = leNet2
# for fichier in ../UCR_TS_Archive_2015/*
# do
#     echo $(basename $fichier .${fichier##*.})
#     echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
#     th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.05 -iter 350
# done


# model = mcdcnn
for fichier in ../UCR_TS_Archive_2015/*
do
    echo $(basename $fichier .${fichier##*.})
    echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
    th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.05 -iter 300
done
