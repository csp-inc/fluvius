Search.setIndex({docnames:["bin/01-usgs-station-acquire","bin/02-preprocess-data","bin/03-image-join","bin/04-data-merge","bin/05-prep-qa-chip-dataset","bin/06a-download-chips-for-qa","bin/06b-upload-good-chips-list","bin/07-remove-bad-obs","bin/08-partition-data","bin/09-MLP-grid-search","bin/10-compile-grid-search-results","bin/11-fit-top-model","bin/12-prediction-inputs","bin/13-predict-tabular","bin/14-make-prediction-chips","bin/15-prep-data-for-app","fluvius","index","modules","preamble/00a-background","preamble/00b-project-goals","preamble/00c-requirements","preamble/00d-quickstart","utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["bin/01-usgs-station-acquire.rst","bin/02-preprocess-data.rst","bin/03-image-join.rst","bin/04-data-merge.rst","bin/05-prep-qa-chip-dataset.rst","bin/06a-download-chips-for-qa.rst","bin/06b-upload-good-chips-list.rst","bin/07-remove-bad-obs.rst","bin/08-partition-data.rst","bin/09-MLP-grid-search.rst","bin/10-compile-grid-search-results.rst","bin/11-fit-top-model.rst","bin/12-prediction-inputs.rst","bin/13-predict-tabular.rst","bin/14-make-prediction-chips.rst","bin/15-prep-data-for-app.rst","fluvius.rst","index.rst","modules.rst","preamble/00a-background.rst","preamble/00b-project-goals.rst","preamble/00c-requirements.rst","preamble/00d-quickstart.rst","utils.rst"],objects:{"src.fluvius":{USGS_Station:[16,1,1,""],USGS_Water_DB:[16,1,1,""],WaterData:[16,1,1,""],WaterStation:[16,1,1,""],normalized_diff:[16,3,1,""]},"src.fluvius.USGS_Station":{create_driver:[16,2,1,""],get_url_text:[16,2,1,""],get_water_attribute:[16,2,1,""],get_water_df:[16,2,1,""],get_water_url:[16,2,1,""],process_soup:[16,2,1,""]},"src.fluvius.USGS_Water_DB":{create_driver:[16,2,1,""],get_marker_info:[16,2,1,""],get_station_df:[16,2,1,""],get_url_text:[16,2,1,""],process_soup:[16,2,1,""]},"src.fluvius.WaterData":{generate_map:[16,2,1,""],get_available_station_list:[16,2,1,""],get_source_df:[16,2,1,""],get_station_data:[16,2,1,""],sort_station_data:[16,2,1,""]},"src.fluvius.WaterStation":{build_catalog:[16,2,1,""],check_response:[16,2,1,""],chip_cloud_analysis:[16,2,1,""],drop_bad_usgs_obs:[16,2,1,""],format_time:[16,2,1,""],get_area_of_interest:[16,2,1,""],get_chip_features:[16,2,1,""],get_chip_metadata:[16,2,1,""],get_cloud_filtered_image_df:[16,2,1,""],get_io_lulc_chip:[16,2,1,""],get_scl_chip:[16,2,1,""],get_spectral_chip:[16,2,1,""],get_time_bounds:[16,2,1,""],get_visual_chip:[16,2,1,""],merge_image_df_with_samples:[16,2,1,""],perform_chip_cloud_analysis:[16,2,1,""],upload_local_to_blob:[16,2,1,""],visualize_chip:[16,2,1,""],write_chip_to_local:[16,2,1,""]},"src.utils":{fit_mlp_cv:[23,3,1,""]},src:{fluvius:[16,0,0,"-"],utils:[23,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],"01":[12,17,23],"02":17,"03":[3,4,17],"04":[2,17],"05":[5,17],"06a":[6,17],"06b":[7,17],"07":17,"08":17,"09":[10,17],"1":[0,19,23],"10":[16,17],"100":[4,23],"11":[10,17],"12":[13,14,17],"123":[8,9,10,11,13,14,15,23],"13":17,"14":17,"15":17,"1970":19,"1e":23,"2":[2,3,4,5,6,7,8,12,15,16,23],"20":[16,23],"2003":19,"2006":19,"2009":19,"2011":19,"2013":16,"2014":16,"2015":[12,16],"2016":16,"2017":16,"2018":16,"2019":[16,19],"2020":16,"2021":[12,16],"2022":17,"25":[2,3,4,5,6,7,8,9,10,11,12,13,14,15],"3":16,"31":12,"3rd":16,"4":23,"4000":4,"4096":22,"5":[8,9,10,11,13,14,15,19,23],"500":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],"7":4,"8":[2,3,4,5,6,7,8,15,16,23],"80":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],"8080":22,"8888":22,"95":19,"class":16,"default":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],"do":21,"float":23,"function":[14,21,23],"int":23,"itacai\u00fana":20,"new":[5,19,22],"public":22,"return":[2,16],"short":5,"true":[2,12,23],A:[8,17,22,23],By:19,For:[1,2,12,13,14,21],If:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22],In:17,It:5,Of:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],One:[5,19],The:[8,9,10,11,12,13,14,15,17,19,20,21,22,23],These:[12,14,17],To:21,_:[5,9],_mask:5,_masking_:9,abc:1,abil:19,accept:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],access:22,accompani:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],accord:[5,9],account:21,acquir:[17,21],activ:[21,23],activation_funct:23,actual:22,ad:22,add:22,addit:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],address:22,advanc:20,after:[6,20],aggreat:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],aggreg:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],ai:17,aim:20,al:19,all:[0,2,10,11,16,17,19,21],alloc:22,allow:21,almost:19,alon:19,along:16,alreadi:22,altern:22,amazon:[17,19],an:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23],ana:[2,12,13,14,15],analysi:19,analyt:17,ani:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23],anyth:22,anywher:22,api:2,app:[4,14,17],appear:8,appli:[14,16],applic:[15,19],approach:19,ar:[5,8,9,11,14,16,17,19,21,22],architectur:20,area:19,arg:23,argument:23,around:[2,3,4,5,6,7,8,15],arrai:[0,16],artifici:[19,20],assign:22,associ:[5,15,21,22,23],assum:4,attempt:14,attribut:16,autom:20,azur:[2,5,6,7,10,22,23],b1:16,b2:16,b:[4,22],background:17,bad:17,band:16,base:[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21,22],basin:[19,20],batch:23,batch_siz:23,been:19,below:22,benchmark:20,best:[9,10,11],between:[17,23],bin:[2,3,4,5,6,7,17,23],blade:22,blob:[2,6,7,10,12,16,23],blobnam:16,block:17,bodi:19,bool:23,both:[4,21],branch:17,brazil:17,brazilian:19,brazilsouth:22,buffer:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23],buffer_dist:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23],build:17,build_catalog:16,c:22,calcul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],can:[2,4,5,16,17,19,21,22],catalog:2,catalogu:12,central:20,chang:[17,19],channel:19,characterist:19,check_respons:16,checkpoint:11,chip:[2,3,7,8,9,10,11,12,13,15,17,23],chip_cloud_analyisi:16,chip_cloud_analysi:16,chips_default:16,choic:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],choos:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],cir:5,classif:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],classifi:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],client:16,clone:21,cloud:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,22,23],cloud_thr:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23],cloudapp:22,cnn:19,co:17,code:[8,17],coe:19,collabor:17,collect:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],color:5,com:22,combin:[10,19],command:[22,23],commonli:22,commun:22,compar:20,compil:[15,17],composit:5,comput:[12,16,21,22],concentr:[2,17],concurr:16,confirm:22,connect:23,consequ:19,conserv:17,consum:14,contact:21,contain:[3,16,21,23],content:16,contextu:19,continu:19,convert:19,convolut:19,core:[9,20],correct:4,correspon:5,correspond:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],costa:19,cover:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],creat:[0,8,9,10,11,13,14,15,21,22],create_driv:16,creation:4,critic:19,cross:[8,9,10,11,13,14,15],crossfold:23,csp:21,csv:[0,1,2,10,13],dai:[2,3,4,5,6,7,8,15,23],data:[0,2,4,5,6,7,9,10,11,12,13,14,16,17,19,20,21,22,23],data_sourc:16,data_src:1,datafram:[3,16],dataset:[5,17,20],date:[2,3,4,5,6,7,8,12,15],day_toler:[2,3,4,5,6,7,8,15,16,23],decai:23,decis:19,deep:20,deforest:19,degrad:19,delet:5,depend:[21,22],depict:5,deploi:[14,20],describ:22,design:14,desir:23,detect:19,determin:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],develop:[8,20],dict:23,dictionari:23,difficulti:19,dimens:16,directori:[5,16,17,21],discharg:[16,19],displai:[4,15],distanc:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],distribut:19,disturb:20,divid:8,dn:22,document:[17,21],documet:17,doe:[1,16,22],done:22,download:[6,12,17],draw:14,drop:16,drop_bad_usgs_ob:16,dt:16,dt_list:16,due:19,dure:[6,8,23],dynam:[19,22],e:[5,17,22],each:[3,5,9,10,12,14,16,22,23],earliest:12,easier:5,ecosystem:[19,20],either:8,element:23,enabl:[17,22],end:12,end_dat:12,engin:21,entir:[12,20],environ:[21,22],epoch:23,epsg:16,equal:[16,23],error:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],estim:19,et:19,evalu:[8,20],event:17,everi:[12,13],exampl:[21,22],execut:17,executor:16,exist:[2,20,22],extract:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,21],f:22,fals:[0,1,2,12,16],featur:[2,3,23],fewer:23,field:2,file:[0,5,6,9,13,17,21,22],filenam:[17,22],filesystem:[5,9],filesytem:[5,6],filter:[3,6,7],fine:4,first:[8,22],fit:[8,9,10,13,14,15,17,23],fit_mlp_cv:23,flag:4,flow:0,fluviu:[8,18,20],flux:19,focu:20,fold:[8,9,10,11,13,14,15,23],folder:9,folds_se:9,folium:16,follow:[6,17,21,22,23],forest:[19,20],format_tim:16,found:[17,19],frequenc:19,freshwat:19,from:[4,9,10,11,12,13,14,16,17,19,20,21,22,23],full:4,further:8,futur:[16,20],g:[4,17,22],gamma:4,gener:[4,5,8,9,10,11,12,13,14,15,16,17,22,23],generate_map:16,get:[0,16],get_area_of_interest:16,get_available_station_list:16,get_chip_featur:16,get_chip_metadata:16,get_cloud_filtered_image_df:16,get_io_lulc_chip:16,get_marker_info:16,get_scl_chip:16,get_source_df:16,get_spectral_chip:16,get_station_data:16,get_station_df:16,get_time_bound:16,get_url_text:16,get_visual_chip:16,get_water_attribut:16,get_water_df:16,get_water_url:16,gif:22,github:17,given:[9,22,23],go:[5,9,22],goal:17,good:[7,17],greater:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,19],grid:[8,11,17,20,23],gui:22,guid:[21,22],h:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],ha:21,hard:8,hash:9,have:[16,19,21],health:17,here:[17,21,22],hidden:23,high:[5,6,7],how:9,howev:[19,21],href:16,hrefs_10m:16,hrefs_20m:16,http:22,human:19,hydro:19,hyperparamet:[8,9,10,20,23],hypertext:22,i:[5,22],id:[5,6],ident:22,identifi:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20],imag:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],imageri:[17,19,20],img_meta:16,img_transform:16,impact:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19],improv:19,includ:[6,8,19,20],increas:19,increasingli:19,index:[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15],index_start:0,individu:[9,14,21],infer:14,inform:[9,14,15,17,19],infrar:5,initi:[8,9,10,11,13,14,15,21,23],input:[2,13,14,17],instal:21,instantan:[0,16],instanti:22,instead:21,instituto:17,instruct:21,integ:[8,9,10,11,13,14,15],intellig:[19,20],intend:14,introduct:17,investig:17,involv:8,ip:22,is_brazil:23,itv:[2,12,13,14,15,22],j:17,join:[3,4,17],json:9,jupyt:22,k:[8,9,10,11,13,14,15],kept:5,keygen:22,km:19,l2a:16,l2a_aot:23,l2a_b02:23,l2a_b03:23,l2a_b04:23,l2a_b05:23,l2a_b06:23,l2a_b07:23,l2a_b08:23,l2a_b11:23,l2a_b12:23,l2a_b8a:23,l2a_wvp:23,l:22,lab:17,land:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],landau:17,lat:16,later:[4,6],latest:12,latrubess:19,layer:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],layer_out_neuron:23,lead:17,learn:[19,20,23],learning_r:23,length:23,less:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,19],level:14,librari:21,limit:19,line:23,link:21,linux:[21,22],list:[2,7,16,17,23],live:22,lmp:10,local:[0,1,2,5,6,9,22],local_root_dir:16,local_save_dir:16,localfil:16,localhost:22,log:[16,21],logfil:16,logist:19,lon:16,loss:[9,10,23],luke:17,lulc:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23],m_cloudthr:[5,9],machin:[0,1,2,5,19,22],macosx:22,main:[17,19],make:[17,19],manag:19,mani:9,map:16,mar:17,marker_text:16,mask:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],mask_method1:[16,23],mask_method2:[5,16,23],match:[2,3,4,5,6,7,8,15],max:4,maximum:[4,23],mean:[19,22],mean_solar_azimuth:23,mean_viewing_azimuth:23,mean_viewing_zenith:23,meant:17,measur:[16,19],merg:[2,17],merge_image_df_with_sampl:16,metadata:11,meter:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],method1:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],method2:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],method:[16,23],microsoft:[17,21],min:4,min_water_pixel:23,minim:16,minimum:[4,23],mitig:19,mlp:[17,23],mndwi:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],model:[0,2,5,8,9,10,12,13,14,15,17,20,21,23],modifi:21,modul:18,monitor:[12,17,19,20],more:16,mount:22,multipl:[14,22],must:[12,21,22,23],n:[8,9,10,11,13,14,15],n_fold:[8,9,10,11,13,14,15,23],n_worker:9,name:[22,23],nan:9,natur:[19,22],navig:19,ndvi:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],necessari:14,need:[21,22],network:[19,22],neural:19,neuron:23,never:8,nn:23,non:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],none:16,normal:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],normalized_diff:16,note:[16,22],notebook:22,noth:16,now:22,num_paramet:23,number:[8,9,10,11,13,14,15,19,23],numer:17,ob:17,object:[16,19],observ:[2,4,16,20,23],observatori:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],obtain:[3,13],occur:19,often:19,ommiss:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],onc:22,one:[16,22],onli:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,22],open:[21,22],oper:22,opportun:19,oppos:5,optim:9,option:2,order:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,17],os:22,other:[9,19,20],our:[14,20],out:[0,1,2,20],outperform:19,output:[9,23],over:[8,9,19,20],own:[21,22],page:22,pair:22,panda:16,parallel:9,param:16,paramet:[8,23],pars:10,partit:[3,9,10,11,13,14,15,17,23],partner:17,password:22,pastureland:19,percent:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],perform:[4,5,6,8,9,16,19,20],perform_chip_cloud_analysi:16,phase:8,physic:9,pipelin:[20,21],pixel:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],plan:19,planetari:[12,21],platform:20,pleas:21,plot:16,png:5,poor:[4,5],popul:21,port:22,portal:22,possibl:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],pre:21,predict:[15,17,19,20],prefix:17,prelu:23,prep:[5,17],prepar:[2,4,23],preprocess:[17,20,23],present:19,preview:5,price:21,primari:23,princip:17,print:23,prior:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],privat:22,process:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,21],process_soup:16,processpoolexecutor:16,profil:22,progress:[20,23],project:[17,21],promis:19,properli:21,protocol:22,provid:[17,19,21,22],proxi:19,prrof:22,pseudorandom:[8,9,10,11,13,14,15,23],pub:22,publickeyfilenam:22,purpos:14,push:17,py:17,pystac:16,python3:17,qa:[2,6,17],qa_chip:5,qc:[2,4,5,6],qualiti:[0,4,5,6,7],queri:[2,10,12],quickstart:17,quiet:16,r:4,radiu:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],rate:23,ration:19,raw:3,rebuilt:17,recommend:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],record:6,refit:[10,11],reflect:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],reforest:19,region:19,remain:[5,6],remot:[2,19,20],remov:[2,3,4,5,6,8,9,10,11,12,13,14,15,17],repo:17,repositori:[17,21],repres:[5,17],represent:4,requir:[4,17,22],reset:22,resourc:19,respect:2,respons:[16,20],restor:20,result:[9,14,17,21],retain:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],return_meta_transform:16,rgb:[4,5],rgb_max:4,rgb_min:4,riparian:19,river:[17,19,20],roadwai:19,root:17,rsa:22,run:[1,2,4,5,12,13,14,17,22,23],runoff:19,s:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,22],safe:16,same:21,sampl:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,23],sample_id:16,satellit:[17,19,20],satur:4,save:[5,7,9,10,11,13,16],scene:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],scienc:17,scientist:[4,5],scl:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23],script:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,23],search:[2,3,4,5,6,7,8,11,12,13,14,15,16,17,20,23],secondari:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],section:[17,21],sediment:[2,17,19,20],sedimentolog:19,see:21,seed:[8,9,10,11,13,14,15,23],seen:[8,22],send:21,sens:[2,19,20],sentinel:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23],separ:[12,23],seri:12,serv:17,server:22,servic:21,set:[2,8,9,10,11,12,13,14,15],settlement:19,setup:17,sever:8,share:22,shorten:9,should:[1,2,5,12,13,14,17,22,23],shouldn:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],shown:19,sign:16,signed_url:16,signific:19,similar:22,sinc:19,singl:[3,10],site:[8,12,13,14,15,20],site_no:16,size:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],sleep_tim:16,snetinel:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],so:[5,14,16],some:16,sort_station_data:16,soup:16,sourc:[1,2,3,12,13,14,16,21],space:[2,19,20],specif:22,specifi:17,spectral:19,squar:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],src:[1,2,12,13,14,16,17,23],ssc:13,stac:[2,12],start:[0,12,22,23],start_dat:12,stastist:10,station:[16,17,20],statist:9,step:[2,3,4,5,14],storag:[2,6,7,10,11,12,20,23],storage_opt:[16,23],store:21,str:23,string:23,strongli:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],structur:19,submiss:22,submit:22,subsequ:[2,3,5,11,12],subset:23,success:20,suggest:21,suppli:4,support:19,surpass:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],suspend:[2,17,19,20],swir:5,system:[19,22],t:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,22],tabular:17,technic:[17,22],techniqu:19,tecnologico:17,termin:22,test:[8,23],than:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,23],thei:21,them:[2,13],thi:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,22,23],thr:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],threshold:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],through:[5,19,22],tile:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,23],time:[2,12,19,20,22],tiv:13,toler:[2,3,4,5,6,7,8,15],toni:17,top:[8,10,17,20],torch:23,total:20,tradit:19,train:[4,5,6,7,8,9,10,11,13,14,15,20,23],transfer:[21,22],troubleshoot:[2,12],tss:20,tune:4,turn:22,two:[16,22],type:[16,22],u:17,ubunut:21,ultim:3,understand:[19,20],uniqu:[5,6,22],unix:22,unless:[2,12,22],updat:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],upload:[7,17],upload_local_to_blob:16,url:[2,3,16],us:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23],usag:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],user:[21,22],usernam:22,usg:[2,12,13,14,16,17],usgs_stat:16,usgs_water_db:16,usgsi:[2,12,13,14],util:18,vale:17,valid:[8,9,10,11,13,14,15,23],valu:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],variabl:[21,22],variou:[20,21,22],verbos:[16,23],view:17,vincent:17,virtual:22,visual:[4,14],visualize_chip:16,vm:[5,21,22],wa:[4,14,17],wai:19,want:22,water:[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,23],waterdata:16,watersh:20,waterst:16,wave:5,we:[14,17,21,22],web:[14,15,16,17,22],websit:17,weight:23,weight_decai:23,were:6,when:[4,8,9,10,11,13,14,15,19,22,23],where:22,which:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,21,22],wish:[21,22],withheld:20,within:[19,21,22],work:[20,21],worker:9,workflow:[3,17,22],wrangl:3,write:[0,1,2,4,12],write_chip:16,write_chip_to_loc:16,write_to_csv:16,written:[2,5],year:16,year_rang:16,you:[21,22],your:21,zachmann:17,zero:4},titles:["01-usgs-station-acquire.py","02-preprocess-data.py","03-image-join.py","04-data-merge.py","05-prep-qa-chip-dataset.py","06a-download-chips-for-qa.py","06b-upload-good-chips-list.py","07-remove-bad-obs.py","08-partition-data.py","09-MLP-grid-search.py","10-compile-grid-search-results.py","11-fit-top-model.py","12-prediction-inputs.py","13-predict-tabular.py","14-make-prediction-chips.py","15-prep-data-for-app.py","fluvius module","Fluvius","src","Background","Project goals","Requirements","Quickstart","utils module"],titleterms:{"01":0,"02":1,"03":2,"04":3,"05":4,"06a":5,"06b":6,"07":7,"08":8,"09":9,"10":10,"11":11,"12":12,"13":13,"14":14,"15":15,acquir:0,app:15,argument:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],authent:21,author:17,azur:21,background:19,bad:7,blob:21,build:21,chip:[4,5,6,14],command:17,compil:10,connect:22,credenti:21,data:[1,3,8,15],dataset:4,docker:21,download:5,fit:11,fluviu:[16,17],gen:22,get:17,goal:20,good:6,grid:[9,10],group:21,imag:2,input:12,intern:17,join:2,kei:22,line:17,list:6,machin:21,make:14,merg:3,mlp:9,model:11,modul:[16,23],name:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],ob:7,partit:8,predict:[12,13,14],prep:[4,15],preprocess:1,project:20,py:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],qa:[4,5],quickstart:22,remov:7,requir:21,resourc:21,result:10,run:21,script:17,search:[9,10],secur:22,shell:22,src:18,ssh:22,start:17,station:0,storag:21,tabular:13,top:11,upload:6,usg:0,util:23,via:22,virtual:21}})