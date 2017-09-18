FROM base/archlinux

ADD mirrorlist /etc/pacman.d/mirrorlist
RUN pacman -Sy
RUN pacman -S --noconfirm python-numpy python-tensorflow
