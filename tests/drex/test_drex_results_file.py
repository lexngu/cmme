from cmme.drex.binding import parse_results_file
import os


def test_results_file_reads_file_without_error():
    rf1 = parse_results_file(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-gaussian-D1.mat"))
    rf2 = parse_results_file(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-gaussian-D2.mat"))
    rf3 = parse_results_file(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-gmm-D1.mat"))
    rf4 = parse_results_file(os.path.join(os.path.dirname(__file__), "../sample_files/drex-resultsfile-poisson-D3.mat"))

