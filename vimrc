au InsertEnter * silent execute "!echo -en \<esc>[5 q"
au InsertLeave * silent execute "!echo -en \<esc>[1 q"

set ruler
set autoindent
set number
set shiftwidth=2
set softtabstop=2
set expandtab

set nostartofline

set ignorecase
set smartcase
set wildmenu
set cpoptions +=*cpo-W*
colorscheme desert
set hls
set ic 
set nu 
set mouse=a

inoremap <Char-0x07F> <BS>
nnoremap <Char-0x07F> <BS>
set backspace=2
set backspace=indent,eol,start

set tags+=./tags;
vmap <C-c> "+yi
vmap <C-x> "+c
vmap <C-v> c<ESC>"+p
imap <C-v> <ESC>"+pa

set nobackup
set nowritebackup
set noswapfile
set clipboard=unnamed
noremap % v%
set belloff=all
