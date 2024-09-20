savefile="out/best/abl.log"

# "UCF50" "PUCPR" "ShanghaiTech" "CARPK" "FSC_gdino" "QNRF" "JHU"
for dataset in "FSC_gdino"; do
    if [[ $dataset == "UCF50" ]]; then
        for id in 1 2 3 4 5 0; do
            echo bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
            bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
        done
    elif [[ $dataset == "ShanghaiTech" ]]; then
        for id in "A" "B"; do
            echo bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
            bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
        done 
    else
        echo bash scripts/test.sh -d $dataset | tee -a $savefile
        bash scripts/test.sh -d $dataset | tee -a $savefile
    fi
done
