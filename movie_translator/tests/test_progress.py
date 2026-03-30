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

    def test_backward_compat_single_arg_set_stage(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_stage('identify')  # old API
        state = tracker._active_files['ep01']
        assert state.current_stage == 'identify'
