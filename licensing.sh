#!/bin/bash

for i in $(find . -name '*.py');
do
  if ! grep -q Copyright $i
  then
    cat LICENSE.header $i >$i.new && mv $i.new $i
  fi
done
