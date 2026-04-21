FROM archlinux:latest
LABEL maintainer='jarrett.aiken@achl.fr'

# Update the base system and install build tooling from official repos.
RUN \
  pacman -Syu --noconfirm --needed \
  archlinux-keyring \
  base-devel \
  fish \
  git \
  git-lfs \
  github-cli \
  openssl \
  rust \
  sudo

# Build and install paru directly from the AUR instead of depending on Chaotic.
RUN \
  useradd -m codespace -p $(openssl passwd -5 "password") \
  && git clone https://aur.archlinux.org/paru.git /tmp/paru \
  && chown -R codespace:codespace /tmp/paru \
  && su codespace -c 'cd /tmp/paru && makepkg -s --noconfirm' \
  && pacman -U --noconfirm /tmp/paru/paru-*.pkg.tar.zst \
  && rm -rf /tmp/paru

# Setup user and sudo permissions
RUN \
  chsh -s /usr/bin/fish codespace \
  && usermod -aG wheel codespace \
  && sed -i '/^# %wheel ALL=(ALL:ALL) ALL/s/^# //' /etc/sudoers

USER codespace
WORKDIR /home/codespace
