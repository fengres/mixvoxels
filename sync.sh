#! /bin/sh
pass=lxh970812
expect -c "
    spawn rsync -av ./ lxh@166.111.131.98:/data/machine/mixvoxels_pub/
    expect {
            *assword {set timeout 300; send $pass\r; exp_continue;}
    }
    exit
    "
