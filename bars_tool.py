# Written by MediaMoots

import sys
import glob
import PySimpleGUI as sg
from totk_audio_classes import Bars
from tkinter import filedialog as fd

### pyinstaller bars_tool.py --onefile

# Functions
def show_options():
    sg.theme('Black')
    event, values = sg.Window('TOTK BARS Tool',
                              [[sg.Text('Options'),
                                sg.Checkbox(default=False, text='Clear BARS?', size=(20, 3), key='ClrB')],
                               [sg.Button('Start Tool'), sg.Button('Cancel')]]).read(close=True)

    if event == 'Start Tool':
        run_bars_tool(values['ClrB'])
    else:
        sys.exit()
   
def run_bars_tool(is_clear_bars):
    bars_path, bwav_path = get_required_paths()

    bars = Bars(bars_path)
    
    if is_clear_bars:
        clear_bars(bars)
        
    add_or_replace_bars(bars, bwav_path)
    
    save_bars(bars)
  
def get_required_paths():
    # bars
    if len(sys.argv) > 1:
        bars_path = sys.argv[1]
    else:
        bars_path = fd.askopenfilename(title="Open BARS file to use...", filetypes=[
                                       ('BARS Files', '*.bars')])
        
    if bars_path == '':
        sys.exit()

    # bwav
    if len(sys.argv) > 2:
        bwav_path = sys.argv[2]
    else:
        bwav_path = fd.askdirectory(
            title="Open folder containing BWAV files...")
        
    if bwav_path == '':
        sys.exit()

    return bars_path, bwav_path

def clear_bars(bars):
    bars.meta_count = 0
    
    bars.metas.clear()
    bars.meta_offsets.clear()
    
    bars.assets.clear()
    bars.asset_offsets.clear()
    
    bars.crc_hashes.clear()
    
    bars.unknown = 4 * b'\x00'
    
    bars.size = bars.get_size()

def add_or_replace_bars(bars, bwavs_path):
    bwav_paths = glob.glob(bwavs_path + "/" + "*.bwav")    
    for bwav_path in bwav_paths:
        bars.add_or_replace_bwav(bwav_path, True)

def save_bars(bars):
    if len(sys.argv) > 3:
        bars_out_path = sys.argv[3]
    else:
        bars_out_path = fd.asksaveasfilename(
            title="Select where to save the new BARS file...", filetypes=[('BARS File', '*.bars')])
        
    if bars_out_path == '':
        sys.exit()

    if bars_out_path is not None:
        bars.write(bars_out_path)
    
# Program Main
if __name__ == '__main__':
    show_options()
