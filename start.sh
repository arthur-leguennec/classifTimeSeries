### Test script

# model = leNet2
# for fichier in ../UCR_TS_Archive_2015/*
# do
#     echo $(basename $fichier .${fichier##*.})
#     echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
#     th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model leNet2 -lr 0.05 -iter 350
# done

th main.lua -model leNet2 -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/ECG5000/ -script -miniBatchSize 10
th main.lua -model mcdcnn -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/ECG5000/ -script -miniBatchSize 10
th main.lua -model leNet2 -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/ECG200/ -script
th main.lua -model mcdcnn -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/ECG200/ -script
th main.lua -model leNet2 -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/Beef/ -script
th main.lua -model mcdcnn -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/Beef/ -script
th main.lua -model leNet2 -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/Adiac/ -script -miniBatchSize 5
th main.lua -model mcdcnn -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/Adiac/ -script -miniBatchSize 5
th main.lua -model leNet2 -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/FordA/ -script -miniBatchSize 15
th main.lua -model mcdcnn -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/FordA/ -script -miniBatchSize 15
th main.lua -model leNet2 -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/FaceAll/ -script -miniBatchSize 10
th main.lua -model mcdcnn -iter 150 -lr 0.025 -lrd 0.02 -pathData ../UCR_TS_Archive_2015/FaceAll/ -script -miniBatchSize 10


# model = mcdcnn
# for fichier in ../UCR_TS_Archive_2015/*
# do
#     echo $(basename $fichier .${fichier##*.})
#     echo "../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})"
#     th main.lua -pathData ../UCR_TS_Archive_2015/$(basename $fichier .${fichier##*.})/ -script -model mcdcnn -lr 0.05 -iter 300
# done
