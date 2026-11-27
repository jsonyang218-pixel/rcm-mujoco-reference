"""Task-local copy of paired collect(): only AB setup and truthful role/target labels added."""
def campaign_collect(spec):
    import numpy as np
    from PIL import Image
    from rcm.scene_v9 import RCMSceneV9
    from lerobot_training.eval_policy import PolicyEvaluator, TASK_TO_DEG_DEG_MM
    from scripts.research.homog_replay_common import make_teacher
    unit = np.asarray([np.pi/180,np.pi/180,.001])
    folder = OUT/'data'/spec['id']
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'rgb').mkdir()
    scene = None
    sensor = None
    rows = []
    actions = []
    previous = np.zeros(3)
    started = time.monotonic()
    try:
        save(folder/'task_status.json', dict(status='running', stage='fresh_scene_setup', pid=os.getpid(), branch=spec))
        scene = RCMSceneV9()
        task = next(t['task'] for t in read(OUT/'experiment_card.json')['targets'] if t['task']['id']==spec['target'])
        world = np.asarray(scene.robot_to_world(np.asarray(task['target_robot_mm'])))
        np.testing.assert_allclose(world, task['target_world_mm'], atol=1e-9, rtol=0)
        scene.set_target_world(world)
        q = scene.target_task(validate=True).copy()
        q[2] += spec['initial_C_offset_m']
        q[:2] += np.asarray(spec['initial_AB_offset_rad'])
        scene.set_task(q)
        scene.enable_motor_execution()
        scene.reset_motor_time(0.)  # Once at FRESH branch setup, never in flight.
        teacher = make_teacher(scene)
        sensor = teacher.sensor
        real_observe = sensor.observe
        cache = {}

        def capture():
            tick_time = float(scene.motor_execution.time_s)
            if cache.get('time') == tick_time:
                return cache['obs'], cache['row']
            sensor.observation_time_s = tick_time
            obs = real_observe()
            rgb = np.asarray(obs['rgb'], np.uint8)
            oct_feature = np.zeros(4, np.float32)
            navigation = obs.get('oct_navigation_depth')
            delta = None if navigation is None else navigation.get('delta_xyz_m')
            if delta is not None and len(delta)==3 and np.isfinite(delta).all():
                oct_feature[:3] = np.asarray(delta,dtype=np.float64)*1000
                oct_feature[3] = 1.
            image_path = folder/'rgb'/f'{len(rows):04d}.png'
            Image.fromarray(rgb).save(image_path, compress_level=1)
            tip = scene.site_mm('needle_tip').copy()
            row = dict(target_id=spec['target'], role=spec['role'], index=len(rows), time_s=tick_time, rgb_path=str(image_path.relative_to(OUT)), rgb_sha256=sha(image_path),
                RGB_shape=list(rgb.shape), RGB_dtype=str(rgb.dtype), actual_sensor_observe_call=len(rows)+1,
                OCT_metadata=obs.get('oct_metadata'),RGB_depth_source=obs.get('oct_depth_source'),
                state_deg_deg_mm=np.asarray(scene.current_task())*TASK_TO_DEG_DEG_MM,
                signed_OCT_mm=oct_feature, actual_velocity_si=np.asarray(scene.motor_execution.actual_velocity).copy(),
                previous_accepted_deg_deg_mm=previous.copy(), encoder_ABCE=list(scene.motor_execution.backend.robot),
                needle_site_world_mm=tip, target_world_mm=world.copy(), true_error_um=float(np.linalg.norm(tip-world)*1000))
            row = json.loads(json.dumps(row, default=default))
            rows.append(row)
            append(folder/'observations.jsonl', row)
            cache.update(time=tick_time, obs=obs, row=row)
            return obs, row

        # One real observation per distinct motor timestamp. The post observation
        # is also the next pre observation, not a second/duplicated frame.
        sensor.observe = lambda: capture()[0]
        capture()
        for raw in spec['prefix_actions_deg_deg_mm']:
            _,pre = capture()
            action = np.asarray(raw)
            scene.apply_prepared_motor(scene.motor_execution.preview(scene.current_task()+action*unit, .05))
            previous = action.copy()
            _,post = capture()
            row = dict(phase='prefix', global_action=len(actions), teacher_step=None,
                pre_observation=pre['index'], post_observation=post['index'], raw_deg_deg_mm=action,
                clipped_deg_deg_mm=action, accepted_deg_deg_mm=action, safety_scale=1., motor_record=scene.motor_execution.last_record)
            row = json.loads(json.dumps(row,default=default))
            actions.append(row)
            append(folder/'actions.jsonl',row)
        prefix_previous = previous.copy()
        pending = {}

        class TeacherEvaluator(PolicyEvaluator):
            def __init__(self):
                self.use_oct = True
                self.measure_inference = False
                self.controller_variant = 'feedback'
                self.fine_evaluator = None
                self.last_terminal_solution = None
                self.terminal_locked_target_world_mm = None

            def reset_buffer(self, seed=None):
                self.previous = prefix_previous.copy()
                self.control_mode_counts = {'teacher_diagnostic':0}

            def record_applied_action(self, action):
                nonlocal previous
                previous = np.asarray(action).copy()
                self.previous = previous.copy()

            def invalidate_action_chunk(self):
                pass

            def get_action(self, rgb, state_abc, oct_feature):
                obs,pre = capture()
                assert oct_feature[3]>.5
                np.testing.assert_array_equal(oct_feature,pre['signed_OCT_mm'])
                np.testing.assert_array_equal(state_abc,pre['state_deg_deg_mm'])
                result = teacher.geometry_action_from_observation(np.asarray(state_abc)*unit, obs,
                    previous_velocity=scene.motor_execution.actual_velocity.copy(), command_q=np.asarray(state_abc)*unit)
                assert result['valid'], f'Teacher observation invalid: {spec["id"]}'
                pending.update(pre=pre['index'], raw=np.asarray(result['action_deg_deg_mm']).copy())
                self.control_mode_counts['teacher_diagnostic'] += 1
                return pending['raw']

        e = TeacherEvaluator()
        def observer(event):
            _,post = capture()  # Real final post RGB/OCT/XYZ is saved as well.
            row = dict(target_id=spec['target'], role=spec['role'], label='same-tick accepted ABC', phase='teacher', global_action=len(actions), teacher_step=event['step'],
                pre_observation=pending['pre'], post_observation=post['index'], raw_deg_deg_mm=pending['raw'],
                clipped_deg_deg_mm=event['clipped_action_deg_deg_mm'], accepted_deg_deg_mm=event['accepted_action_deg_deg_mm'],
                safety_scale=event['safety_scale'], motor_record=event['motor_record'])
            row = json.loads(json.dumps(row,default=default))
            actions.append(row)
            append(folder/'actions.jsonl',row)
        e.step_observer = observer
        ep = e.evaluate_episode(scene, world, max_steps=spec['teacher_steps'], error_threshold_mm=.00035,
            control_period_s=.05, action_limit_deg_deg_mm=(.2,.1,.03), max_consecutive_safety_limited=None,
            seed=434300+int(spec['target'][1:]), sensor=sensor, stop_on_success=False, progress_every=80)
        save(folder/'teacher_trajectory.json',ep)
        assert ep['early_stop_reason'] is None and len(actions)==2+spec['teacher_steps']
        assert len(rows)==3+spec['teacher_steps']
        result = dict(status='completed', time=now(), branch=spec, observations=len(rows), teacher_actions=spec['teacher_steps'],
            real_sensor_calls=len(rows), final_true_error_um=rows[-1]['true_error_um'], elapsed_s=time.monotonic()-started,
            final_post_RGB_OCT_saved=True, no_midbranch_resets=True, neural_action_calls=0)
        save(folder/'summary.json',result)
        save(folder/'task_status.json',result,update=True)
        record('teacher_branch_completed',result)
    except BaseException as error:
        failure = dict(status='technical_failure', time=now(), branch=spec, error=repr(error), traceback=traceback.format_exc(),
            saved_observations=len(rows), saved_actions=len(actions), scientific_negative=False)
        save(folder/'failure.json',failure)
        save(folder/'task_status.json',failure,update=True)
        record('teacher_branch_failed',failure)
        raise
    finally:
        if sensor is not None:
            sensor.close()
        if scene is not None:
            scene.close()
