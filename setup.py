#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib
import matplotlib.pyplot as plt
import shutil
import platform

MODULE_NAME=os.path.basename(__file__)
INSTALL_DIR=os.path.dirname(os.path.realpath(__file__))
print('INSTALL_DIR',INSTALL_DIR)
print('MODULE_NAME',MODULE_NAME)

SETTING_FONT_NAME='Fantasque Sans Mono'

if platform.system()=='Linux':
    print('Installing fonts')
    if not os.path.exist('~/.fonts'): os.mkdir('~/.fonts')

    assert(os.system('cp %s/fantasque-sans-mono/TTF/FantasqueSansMono-*.ttf ~/.fonts'%(INSTALL_DIR))==0)

    assert(os.system('fc-cache -f -v')==0)

    if input('This will remove the matplotlib cache located at '+matplotlib.get_cachedir()+' (y/N)? ').strip().lower()!='y':
        raise Exception('Setup cancelled')
    shutil.rmtree(matplotlib.get_cachedir())

    print('Done')
else:
    print('No setup script is yet available for this platform, please continue manually and seek assistance if necessary')
