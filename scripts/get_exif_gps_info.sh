for i in `ls -1`; do (echo "$i";exif --no-fixup $i |egrep -i 'gps|lat|lon');done >exif_gps_info.txt
