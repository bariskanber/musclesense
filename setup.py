#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib
import matplotlib.pyplot as plt
import shutil

MODULE_NAME=os.path.basename(__file__)
INSTALL_DIR=os.path.dirname(os.path.realpath(__file__))
print('INSTALL_DIR',INSTALL_DIR)
print('MODULE_NAME',MODULE_NAME)

SETTING_FONT_NAME='Fantasque Sans Mono'

available_fonts=[f.name for f in matplotlib.font_manager.fontManager.ttflist]
if SETTING_FONT_NAME not in available_fonts:
    raise Exception('Please install the required font "%s" available at %s and try again'%(SETTING_FONT_NAME,INSTALL_DIR))

if input('This will remove the matplotlib cache located at '+matplotlib.get_cachedir()+' (y/N)? ').strip().lower()!='y':
    raise Exception('Setup cancelled')
shutil.rmtree(matplotlib.get_cachedir())

print('Done')