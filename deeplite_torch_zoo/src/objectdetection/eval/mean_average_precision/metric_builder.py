"""
MIT License

Copyright (c) 2020 Sergei Belousov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .adapter import AdapterDefault
from .mean_average_precision_2d import MeanAveragePrecision2d
from .multiprocessing import MetricMultiprocessing

metrics_dict = {
    'map_2d': MeanAveragePrecision2d
}

class MetricBuilder:
    @staticmethod
    def get_metrics_list():
        """ Get evaluation metrics list."""
        return list(metrics_dict.keys())

    @staticmethod
    def build_evaluation_metric(metric_type, async_mode=False, adapter_type=AdapterDefault, *args, **kwargs):
        """ Build evaluation metric.

        Arguments:
            metric_type (str): type of evaluation metric.
            async_mode (bool): use multiprocessing metric.
            adapter_type (AdapterBase): type of adapter class.

        Returns:
            metric_fn (MetricBase): instance of the evaluation metric.
        """
        assert metric_type in metrics_dict, "Unknown metric_type"
        if not async_mode:
            metric_fn = metrics_dict[metric_type](*args, **kwargs)
        else:
            metric_fn = MetricMultiprocessing(metrics_dict[metric_type], *args, **kwargs)
        return adapter_type(metric_fn)
