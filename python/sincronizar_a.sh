#!/bin/bash


host=capra.dsic.upv.es

while [ $# -ge 1  ] 
do
    case $1 in
        --host) host=$2 ; shift ;;
        *) echo "Wrong option: $1 " ;;
    esac
    shift
done



rsync -avHCcn --delete --exclude "*.pyc" --exclude "mypythonlib.tgz" ./   jon@${host}:python/

OPT="NO"

echo ""
echo " The target computer is ${host} "
echo -n "Do you want to go with the syncronization? (YES/NO): "
read opt

if [ "X${opt}" == "XYES" ]
then
    rsync -avHCc --delete --exclude "*.pyc" --exclude "mypythonlib.tgz" ./   jon@${host}:python/
fi
