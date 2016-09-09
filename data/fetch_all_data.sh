for dir in ./*/
do
    echo "Fetching data for directory: " $dir
    cd $dir
    sh get_data.sh 2>&1 > /dev/null
    cd ..
done
