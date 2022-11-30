import pandas as pd
import argparse

def parse_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    psnr = float(lines[1])
    ssim = float(lines[3])
    dssim = (1-ssim)/2
    lpips = float(lines[4])
    flip = float(lines[6])
    jod = float(lines[7])
    return {
        'PSNR': psnr,
        'DSSIM': dssim,
        'LPIPS': lpips,
        'FLIP': flip,
        'JOD': jod
    }

def parse_txts(txt_paths):
    names = []
    res = []
    for path in txt_paths:
        if 'coffee_martini' in path:
            names.append('coffee-martini')
        elif 'salmon' in path:
            names.append('flame-salmon')
        elif 'spinch' in path or 'spinach' in path:
            names.append('cook-spinach')
        elif 'beef' in path:
            names.append('cut_roasted_beef')
        elif 'steak' in path and 'sear' not in path:
            names.append('flame-steak')
        elif 'sear' in path:
            names.append('sear-steak')
        else:
            raise NotImplementedError
    for txt_path in txt_paths:
        res.append(parse_txt(txt_path))
    # metrics = ['PSNR', 'DSSIM', 'LPIPS', 'FLIP', 'JOD']
    PSNRs = [r['PSNR'] for r in res]
    DSSIMs = [r['DSSIM'] for r in res]
    LPIPSs = [r['LPIPS'] for r in res]
    FLIPs = [r['FLIP'] for r in res]
    JODs = [r['JOD'] for r in res]
    df = pd.DataFrame({
        'DATA': names + ['mean'],
        'PSNR': PSNRs + [sum(PSNRs)/len(PSNRs)],
        'DSSIM': DSSIMs + [sum(DSSIMs)/len(DSSIMs)],
        'LPIPS': LPIPSs + [sum(LPIPSs)/len(LPIPSs)],
        'FLIP': FLIPs + [sum(FLIPs)/len(FLIPs)],
        'JOD': JODs + [sum(JODs)/len(JODs)],
    })
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+')
    args = parser.parse_args()
    parse_txts(args.files)