from movie_translator.progress import FileState, ProgressTracker


class TestFileState:
    def test_initial_state(self):
        state = FileState(name='ep01', start_time=0.0)
        assert state.current_stage == ''
        assert state.stages_done == []
        assert state.gpu_status == 'none'


class TestProgressTrackerMultiFile:
    def test_start_file_adds_to_active(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        assert 'ep01' in tracker._active_files

    def test_complete_file_removes_from_active(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.complete_file('ep01', 'success')
        assert 'ep01' not in tracker._active_files
        assert tracker._completed == 1

    def test_multiple_files_active(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.start_file('ep02')
        assert len(tracker._active_files) == 2

    def test_set_stage_updates_file(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_stage('ep01', 'identify')
        tracker.set_stage('ep01', 'fetch')
        state = tracker._active_files['ep01']
        assert state.current_stage == 'fetch'
        assert 'identify' in state.stages_done

    def test_set_gpu_status(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_gpu_status('ep01', 'queued')
        assert tracker._active_files['ep01'].gpu_status == 'queued'

    def test_set_stage_progress_new_api(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_stage('ep01', 'translate')
        tracker.set_stage_progress('ep01', 50, 200, rate=4.2)
        state = tracker._active_files['ep01']
        assert state.stage_progress == (50, 200, 4.2)

    def test_backward_compat_single_arg_set_stage(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_stage('identify')  # old API
        state = tracker._active_files['ep01']
        assert state.current_stage == 'identify'


class TestGpuWorkerPanel:
    def test_gpu_task_started_sets_state(self):
        tracker = ProgressTracker(total_files=5)
        tracker.gpu_task_started('translate', 'ep01')
        assert tracker._gpu.current_task_type == 'translate'
        assert tracker._gpu.current_file == 'ep01'

    def test_gpu_task_completed_clears_and_records(self):
        tracker = ProgressTracker(total_files=5)
        tracker.gpu_task_started('translate', 'ep01')
        tracker.gpu_task_completed('translate', 'ep01')
        assert tracker._gpu.current_task_type == ''
        assert tracker._gpu.current_file == ''
        assert len(tracker._gpu.recent) == 1
        assert 'Translated' in tracker._gpu.recent[0]
        assert 'ep01' in tracker._gpu.recent[0]

    def test_gpu_task_failed_records_failure(self):
        tracker = ProgressTracker(total_files=5)
        tracker.gpu_task_started('ocr', 'ep02')
        tracker.gpu_task_failed('ocr', 'ep02')
        assert tracker._gpu.current_task_type == ''
        assert len(tracker._gpu.recent) == 1
        assert '✗' in tracker._gpu.recent[0]

    def test_gpu_task_progress_updates(self):
        tracker = ProgressTracker(total_files=5)
        tracker.gpu_task_started('translate', 'ep01')
        tracker.gpu_task_progress(50, 200, 4.5)
        assert tracker._gpu.progress == (50, 200, 4.5)

    def test_gpu_queue_size(self):
        tracker = ProgressTracker(total_files=5)
        tracker.gpu_queue_size(3)
        assert tracker._gpu.queue_depth == 3
