# Copyright (C) 2005 The T2 SDE Project
# Copyright (C) XXXX - 2005 Debian
# GNU GPLv2
find_free_servernum() {
    # Sadly, the "local" keyword is not POSIX.  Leave the next line commented in
    # the hope Debian Policy eventually changes to allow it in /bin/sh scripts
    # anyway.
    #local i

    i=1
    while [ -f /tmp/.X$i-lock ]; do
        i=$(($i + 1))
    done
    echo $i
}
find_free_servernum
