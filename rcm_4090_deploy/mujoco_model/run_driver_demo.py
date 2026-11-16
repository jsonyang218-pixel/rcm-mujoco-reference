"""ROS-free controlSignal -> motor model -> MuJoCo -> Encoder feedback demo.

The command waveform is for simulation visualization, not a hardware experiment.
No networking, serial device, or ROS initialization occurs in this script.
"""
import os
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MUJOCO_GL','egl' if sys.platform=='linux' else 'glfw')
os.environ.setdefault('MESA_SHADER_CACHE_DIR',str(HERE/'.cache'))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
import argparse,json,csv,math
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from driver_loader import load_driver
from motor_backend import MotorBackend
from model_sync import DriverModel

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('new','legacy'),default='new')
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    parser.add_argument('--handles',type=int,nargs=4,help='Single cumulative ABCE handle target from the supplied session origin')
    parser.add_argument('--initial-handles',type=int,nargs=4,default=(0,0,0,0))
    args=parser.parse_args();out=args.output_dir;out.mkdir(exist_ok=True,parents=True)
    kin=load_driver()
    from robot_control.response_signals import WireSineSignal
    signal=WireSineSignal(frequency_hz=.05,cycles=1,baseline_s=1,recovery_s=1)
    duration=2.0 if args.handles else signal.duration_s
    backend=MotorBackend(needle_profile=args.profile,initial_handles=tuple(args.initial_handles))
    display=DriverModel(needle_profile=args.profile)
    display.export_xml(HERE/f'model_{args.profile}.xml')
    if args.profile=='new':display.export_xml(HERE/'model.xml')
    frames=[];rows=[];events=[];maximum_error=0.
    with mujoco.Renderer(display.model,height=600,width=900) as renderer:
        for index in range(round(duration/.01)+1):
            now=index*.01
            if index%5==0:
                scalar=signal.at(now)[0]
                # Deliberately visible simulation signal; original experimental
                # WireSineExperiment amplitude limits are not altered or invoked.
                handles=tuple(args.handles) if args.handles else tuple(
                    origin+round(a*scalar) for origin,a in zip(args.initial_handles,(300,700,900,0)))
                command=kin.make_control_message(handles,'hold')
                backend.consume(command,now)
                events.append({'time_s':now,'handles':handles,'stepForward':False,'stepBackward':False})
            feedback=backend.step(.01,now)
            diagnostics=display.sync(backend.robot)
            feedback=display.feedback_from_model(feedback)
            maximum_error=max(maximum_error,diagnostics['max_fk_error_m'])
            assert diagnostics['max_fk_error_m']<1e-9
            rows.append({'time_s':now,**{f'handle_{a}':int(getattr(feedback,'handleEncoder'+a)) for a in 'ABCE'},
                **{f'robot_{a}':int(getattr(feedback,'robotEncoder'+a)) for a in 'ABCE'},
                **{f'junction_{a}_mm':float(getattr(feedback,'pos'+a)) for a in 'XYZ'},
                'A_deg':diagnostics['A_deg'],'B_relative_deg':diagnostics['B_relative_deg'],'C_axial_mm':diagnostics['C_axial_mm']})
            if index%40==0 or index==round(duration/.01):
                renderer.update_scene(display.data,camera='overview');im=Image.fromarray(renderer.render().copy())
                draw=ImageDraw.Draw(im);draw.rectangle((0,0,900,82),fill=(20,27,36))
                draw.text((12,8),f'v9 | integral silver cover | pitch correction {display.installation["pitch_y_deg"]:+.0f}deg | t={now:5.2f}s',fill='white')
                draw.text((12,27),'controlSignal cumulative handles ABCE: '+str(backend.handles),fill='#87e7ef')
                draw.text((12,44),'Encoder robot counts ABCE: '+str(backend.robot),fill='#d7dbdf')
                draw.text((12,62),f'A={diagnostics["A_deg"]:+.3f}deg  Brel={diagnostics["B_relative_deg"]:+.3f}deg  Caxial={diagnostics["C_axial_mm"]:+.4f}mm',fill='#f0c989')
                frames.append(im)
    frames[0].save(out/'driver_motion.gif',save_all=True,append_images=frames[1:],duration=100 if not args.handles else 400,loop=0)
    frames[len(frames)//3].save(out/'driver_preview.png')
    with (out/'driver_trace.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'control_commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    result={'status':'PASS','source_commit':'c2a3ffc5f0801706a48559b3f514b9d32cae0435','profile':args.profile,
        'simulation_duration_s':duration,'integration_steps':len(rows),'accepted_control_packets':len(events),
        'rendered_frames':len(frames),'max_mujoco_vs_driver_fk_error_m':maximum_error,
        'ros_transport_tested':False,'hardware_connected':False,'simulation_dynamics':'chosen first-order motor response, tau=.04s; not identified',
        'feedback_XYZ':'MuJoCo junction, inverse-aligned into driver millimetres, float32',
        'visual_signal':'source WireSineSignal timing with explicitly enlarged handles for display, not the source experimental envelope'}
    (out/'demo_validation.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for ax,name in zip(axes,('A_deg','B_relative_deg','C_axial_mm')):
        ax.plot([r['time_s'] for r in rows],[r[name] for r in rows]);ax.set_ylabel(name);ax.grid(alpha=.3)
    axes[-1].set_xlabel('simulation time (s)');fig.suptitle('Offline protocol motor response; not identified hardware dynamics');fig.tight_layout();fig.savefig(out/'driver_response.png',dpi=140);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
