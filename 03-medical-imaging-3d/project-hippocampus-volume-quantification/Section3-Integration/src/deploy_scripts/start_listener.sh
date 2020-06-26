#!/bin/bash

curl -X POST http://localhost:8042/tools/execute-script --data-binary @route_dicoms.lua -v
# sudo storescp 106 -v -aet HIPPOAI -od /home/workspace/listener --sort-on-study-uid st
sudo /opt/aihcnd-applications/dcmtk-3.6.5-linux-x86_64-static/bin/storescp 106 -v -aet HIPPOAI -od /home/workspace/listener --sort-on-study-uid st