HOME_FOLDER=/root

curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | bash || true
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${HOME_FOLDER}/.oh-my-zsh/custom/themes/powerlevel10k

cp zshrc $HOME_FOLDER/.zshrc
cp p10k.zsh $HOME_FOLDER/.p10k.zsh

