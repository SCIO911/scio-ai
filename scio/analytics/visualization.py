"""
SCIO Analytics Visualization (MEGA-UPGRADE)

Interaktive Charts und Dashboards.

Features:
- Plotly Interactive Charts
- Automatic EDA Reports
- Dashboard Generation
- Export zu PNG/SVG/HTML
"""

from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Union
from pathlib import Path
import numpy as np

from scio.core.logging import get_logger

logger = get_logger(__name__)

# Optional Visualization Libraries
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


class ChartBuilder:
    """
    MEGA-UPGRADE: Interaktive Chart-Erstellung mit Plotly

    Unterstützte Chart-Typen:
    - Line Charts
    - Bar Charts
    - Scatter Plots
    - Histograms
    - Box Plots
    - Heatmaps
    - Pie/Donut Charts
    - Area Charts
    - Treemaps
    """

    def __init__(self, theme: str = "plotly_white"):
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly nicht verfügbar - Charts deaktiviert")

        self.theme = theme
        self._figures: Dict[str, Any] = {}

    def line_chart(
        self,
        x: List[Any],
        y: Union[List[float], Dict[str, List[float]]],
        title: str = "Line Chart",
        x_label: str = "X",
        y_label: str = "Y",
        markers: bool = True,
    ) -> Any:
        """
        Erstellt Linien-Diagramm.

        Args:
            x: X-Achsen-Werte
            y: Y-Werte (single list oder dict für multi-line)
            title: Chart-Titel
            x_label: X-Achsen-Beschriftung
            y_label: Y-Achsen-Beschriftung
            markers: Datenpunkte anzeigen

        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        if isinstance(y, dict):
            for name, values in y.items():
                fig.add_trace(go.Scatter(
                    x=x, y=values, name=name,
                    mode='lines+markers' if markers else 'lines'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers' if markers else 'lines'
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
        )

        return fig

    def bar_chart(
        self,
        categories: List[str],
        values: Union[List[float], Dict[str, List[float]]],
        title: str = "Bar Chart",
        orientation: str = "v",
        stacked: bool = False,
    ) -> Any:
        """
        Erstellt Balken-Diagramm.

        Args:
            categories: Kategorie-Namen
            values: Werte (single list oder dict für grouped)
            title: Chart-Titel
            orientation: 'v' (vertikal) oder 'h' (horizontal)
            stacked: Gestapelt anzeigen

        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        if isinstance(values, dict):
            for name, vals in values.items():
                if orientation == 'h':
                    fig.add_trace(go.Bar(y=categories, x=vals, name=name, orientation='h'))
                else:
                    fig.add_trace(go.Bar(x=categories, y=vals, name=name))
        else:
            if orientation == 'h':
                fig.add_trace(go.Bar(y=categories, x=values, orientation='h'))
            else:
                fig.add_trace(go.Bar(x=categories, y=values))

        barmode = 'stack' if stacked else 'group'
        fig.update_layout(title=title, barmode=barmode, template=self.theme)

        return fig

    def scatter_plot(
        self,
        x: List[float],
        y: List[float],
        color: List[Any] = None,
        size: List[float] = None,
        labels: List[str] = None,
        title: str = "Scatter Plot",
        x_label: str = "X",
        y_label: str = "Y",
        trendline: bool = False,
    ) -> Any:
        """
        Erstellt Streudiagramm.

        Args:
            x: X-Werte
            y: Y-Werte
            color: Farbkodierung
            size: Punktgrößen
            labels: Punkt-Labels
            title: Chart-Titel
            trendline: Trendlinie anzeigen

        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                color=color,
                size=size or 10,
                colorscale='Viridis' if color else None,
                showscale=True if color else False,
            ),
            text=labels,
            hoverinfo='text+x+y' if labels else 'x+y',
        ))

        if trendline and len(x) > 1:
            # Lineare Regression für Trendlinie
            x_arr = np.array(x)
            y_arr = np.array(y)
            coeffs = np.polyfit(x_arr, y_arr, 1)
            trend_y = np.polyval(coeffs, x_arr)

            fig.add_trace(go.Scatter(
                x=x, y=trend_y.tolist(),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red'),
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
        )

        return fig

    def histogram(
        self,
        data: List[float],
        bins: int = 30,
        title: str = "Histogram",
        x_label: str = "Value",
        show_kde: bool = False,
    ) -> Any:
        """
        Erstellt Histogramm.

        Args:
            data: Daten
            bins: Anzahl Bins
            title: Chart-Titel
            x_label: X-Achsen-Beschriftung
            show_kde: Kernel Density Estimate anzeigen

        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            histnorm='probability density' if show_kde else '',
        ))

        if show_kde:
            # Simple KDE approximation
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(min(data), max(data), 100)
                fig.add_trace(go.Scatter(
                    x=x_range, y=kde(x_range),
                    mode='lines',
                    name='KDE',
                ))
            except Exception:
                pass

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title='Density' if show_kde else 'Count',
            template=self.theme,
        )

        return fig

    def box_plot(
        self,
        data: Union[List[float], Dict[str, List[float]]],
        title: str = "Box Plot",
        show_points: bool = True,
    ) -> Any:
        """
        Erstellt Box-Plot.

        Args:
            data: Daten (single list oder dict für multiple)
            title: Chart-Titel
            show_points: Datenpunkte anzeigen

        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        if isinstance(data, dict):
            for name, values in data.items():
                fig.add_trace(go.Box(
                    y=values, name=name,
                    boxpoints='all' if show_points else False,
                ))
        else:
            fig.add_trace(go.Box(
                y=data,
                boxpoints='all' if show_points else False,
            ))

        fig.update_layout(title=title, template=self.theme)

        return fig

    def heatmap(
        self,
        data: List[List[float]],
        x_labels: List[str] = None,
        y_labels: List[str] = None,
        title: str = "Heatmap",
        colorscale: str = "Viridis",
        annotate: bool = True,
    ) -> Any:
        """
        Erstellt Heatmap.

        Args:
            data: 2D-Daten
            x_labels: X-Achsen-Labels
            y_labels: Y-Achsen-Labels
            title: Chart-Titel
            colorscale: Farbskala
            annotate: Werte annotieren

        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            texttemplate='%{z:.2f}' if annotate else None,
            textfont={"size": 10},
        ))

        fig.update_layout(title=title, template=self.theme)

        return fig

    def pie_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str = "Pie Chart",
        hole: float = 0,
    ) -> Any:
        """
        Erstellt Kreis-/Donut-Diagramm.

        Args:
            labels: Kategorie-Namen
            values: Werte
            title: Chart-Titel
            hole: 0 = Pie, 0.3-0.7 = Donut

        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=hole,
        ))

        fig.update_layout(title=title, template=self.theme)

        return fig

    def correlation_matrix(
        self,
        data: np.ndarray,
        feature_names: List[str],
        title: str = "Correlation Matrix",
    ) -> Any:
        """
        Erstellt Korrelationsmatrix.

        Args:
            data: Feature Matrix (n_samples x n_features)
            feature_names: Feature-Namen
            title: Chart-Titel

        Returns:
            Plotly Figure
        """
        corr = np.corrcoef(data.T)

        return self.heatmap(
            data=corr.tolist(),
            x_labels=feature_names,
            y_labels=feature_names,
            title=title,
            colorscale="RdBu_r",
            annotate=True,
        )

    def time_series(
        self,
        dates: List[Any],
        values: Union[List[float], Dict[str, List[float]]],
        title: str = "Time Series",
        range_slider: bool = True,
    ) -> Any:
        """
        Erstellt Zeitreihen-Diagramm.

        Args:
            dates: Zeitstempel
            values: Werte
            title: Chart-Titel
            range_slider: Range-Slider anzeigen

        Returns:
            Plotly Figure
        """
        fig = self.line_chart(dates, values, title=title, x_label="Date", y_label="Value")

        if fig and range_slider:
            fig.update_xaxes(rangeslider_visible=True)

        return fig

    def save(
        self,
        fig: Any,
        path: str,
        format: str = "html",
        width: int = 1200,
        height: int = 800,
    ) -> str:
        """
        Speichert Figure.

        Args:
            fig: Plotly Figure
            path: Ausgabepfad
            format: 'html', 'png', 'svg', 'pdf'
            width: Breite in Pixeln
            height: Höhe in Pixeln

        Returns:
            Gespeicherter Pfad
        """
        if not PLOTLY_AVAILABLE or fig is None:
            return ""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'html':
            fig.write_html(str(path))
        else:
            fig.write_image(
                str(path),
                format=format,
                width=width,
                height=height,
            )

        logger.info(f"Chart gespeichert: {path}")
        return str(path)


class EDAReport:
    """
    MEGA-UPGRADE: Automatic Exploratory Data Analysis Reports

    Generiert:
    - Datenübersicht
    - Statistische Zusammenfassung
    - Verteilungen
    - Korrelationen
    - Fehlende Werte
    """

    def __init__(self, chart_builder: ChartBuilder = None):
        self.charts = chart_builder or ChartBuilder()
        self._report_sections: List[Dict[str, Any]] = []

    def generate(
        self,
        data: np.ndarray,
        feature_names: List[str] = None,
        target_column: int = None,
    ) -> Dict[str, Any]:
        """
        Generiert EDA Report.

        Args:
            data: Feature Matrix
            feature_names: Feature-Namen
            target_column: Index der Zielvariable

        Returns:
            Report als dict
        """
        n_samples, n_features = data.shape
        feature_names = feature_names or [f"Feature_{i}" for i in range(n_features)]

        report = {
            'overview': self._generate_overview(data, feature_names),
            'statistics': self._generate_statistics(data, feature_names),
            'distributions': self._generate_distributions(data, feature_names),
            'correlations': self._generate_correlations(data, feature_names),
            'missing_values': self._check_missing_values(data, feature_names),
        }

        if target_column is not None:
            report['target_analysis'] = self._analyze_target(
                data, feature_names, target_column
            )

        return report

    def _generate_overview(
        self,
        data: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Generiert Datenübersicht."""
        return {
            'n_samples': data.shape[0],
            'n_features': data.shape[1],
            'feature_names': feature_names,
            'memory_mb': round(data.nbytes / 1024 / 1024, 2),
            'dtype': str(data.dtype),
        }

    def _generate_statistics(
        self,
        data: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Generiert statistische Zusammenfassung."""
        stats = {}

        for i, name in enumerate(feature_names):
            col = data[:, i]
            col_clean = col[~np.isnan(col)]

            if len(col_clean) == 0:
                continue

            stats[name] = {
                'count': len(col_clean),
                'mean': float(np.mean(col_clean)),
                'std': float(np.std(col_clean)),
                'min': float(np.min(col_clean)),
                '25%': float(np.percentile(col_clean, 25)),
                '50%': float(np.percentile(col_clean, 50)),
                '75%': float(np.percentile(col_clean, 75)),
                'max': float(np.max(col_clean)),
            }

        return stats

    def _generate_distributions(
        self,
        data: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Generiert Verteilungsinformationen."""
        distributions = []

        for i, name in enumerate(feature_names):
            col = data[:, i]
            col_clean = col[~np.isnan(col)]

            if len(col_clean) == 0:
                continue

            # Berechne Skewness und Kurtosis
            mean = np.mean(col_clean)
            std = np.std(col_clean)

            if std > 0:
                skewness = float(np.mean(((col_clean - mean) / std) ** 3))
                kurtosis = float(np.mean(((col_clean - mean) / std) ** 4) - 3)
            else:
                skewness = 0.0
                kurtosis = 0.0

            distributions.append({
                'feature': name,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': abs(skewness) < 0.5 and abs(kurtosis) < 1,
            })

        return distributions

    def _generate_correlations(
        self,
        data: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Generiert Korrelationsanalyse."""
        # Berechne Korrelationsmatrix
        corr_matrix = np.corrcoef(data.T)

        # Finde starke Korrelationen
        strong_correlations = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = corr_matrix[i, j]
                if not np.isnan(corr) and abs(corr) > 0.7:
                    strong_correlations.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'correlation': float(corr),
                    })

        return {
            'matrix': corr_matrix.tolist(),
            'strong_correlations': strong_correlations,
        }

    def _check_missing_values(
        self,
        data: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Prüft fehlende Werte."""
        missing = {}

        for i, name in enumerate(feature_names):
            col = data[:, i]
            n_missing = int(np.sum(np.isnan(col)))
            if n_missing > 0:
                missing[name] = {
                    'count': n_missing,
                    'percentage': round(n_missing / len(col) * 100, 2),
                }

        return {
            'features_with_missing': missing,
            'total_missing': sum(m['count'] for m in missing.values()),
        }

    def _analyze_target(
        self,
        data: np.ndarray,
        feature_names: List[str],
        target_column: int,
    ) -> Dict[str, Any]:
        """Analysiert Zielvariable."""
        target = data[:, target_column]
        target_clean = target[~np.isnan(target)]

        unique_values = np.unique(target_clean)

        if len(unique_values) <= 10:
            # Klassifikation
            value_counts = {}
            for v in unique_values:
                value_counts[str(v)] = int(np.sum(target_clean == v))

            return {
                'type': 'classification',
                'n_classes': len(unique_values),
                'class_distribution': value_counts,
                'is_balanced': max(value_counts.values()) / min(value_counts.values()) < 3,
            }
        else:
            # Regression
            return {
                'type': 'regression',
                'mean': float(np.mean(target_clean)),
                'std': float(np.std(target_clean)),
                'range': [float(np.min(target_clean)), float(np.max(target_clean))],
            }

    def to_html(self, report: Dict[str, Any], output_path: str) -> str:
        """
        Exportiert Report als HTML.

        Args:
            report: Generierter Report
            output_path: Ausgabepfad

        Returns:
            Pfad zur HTML-Datei
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>SCIO EDA Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #2c3e50; }",
            "h2 { color: #3498db; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #3498db; color: white; }",
            ".metric { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }",
            "</style>",
            "</head><body>",
            "<h1>SCIO - Exploratory Data Analysis Report</h1>",
        ]

        # Overview
        overview = report.get('overview', {})
        html_parts.append("<h2>Overview</h2>")
        html_parts.append("<div class='metric'>")
        html_parts.append(f"<p><strong>Samples:</strong> {overview.get('n_samples', 'N/A')}</p>")
        html_parts.append(f"<p><strong>Features:</strong> {overview.get('n_features', 'N/A')}</p>")
        html_parts.append(f"<p><strong>Memory:</strong> {overview.get('memory_mb', 'N/A')} MB</p>")
        html_parts.append("</div>")

        # Statistics Table
        stats = report.get('statistics', {})
        if stats:
            html_parts.append("<h2>Statistics</h2>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>")
            for name, s in stats.items():
                html_parts.append(f"<tr><td>{name}</td>")
                html_parts.append(f"<td>{s['mean']:.4f}</td>")
                html_parts.append(f"<td>{s['std']:.4f}</td>")
                html_parts.append(f"<td>{s['min']:.4f}</td>")
                html_parts.append(f"<td>{s['max']:.4f}</td></tr>")
            html_parts.append("</table>")

        # Missing Values
        missing = report.get('missing_values', {})
        if missing.get('total_missing', 0) > 0:
            html_parts.append("<h2>Missing Values</h2>")
            html_parts.append(f"<p>Total missing: {missing['total_missing']}</p>")

        # Correlations
        corr = report.get('correlations', {})
        strong = corr.get('strong_correlations', [])
        if strong:
            html_parts.append("<h2>Strong Correlations</h2>")
            html_parts.append("<ul>")
            for c in strong:
                html_parts.append(
                    f"<li>{c['feature_1']} <-> {c['feature_2']}: {c['correlation']:.3f}</li>"
                )
            html_parts.append("</ul>")

        html_parts.append("</body></html>")

        # Speichern
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(html_parts))

        return str(output_path)


class DashboardBuilder:
    """
    MEGA-UPGRADE: Dashboard Generator

    Erstellt interaktive Dashboards mit mehreren Charts.
    """

    def __init__(self, title: str = "SCIO Dashboard"):
        self.title = title
        self.charts = ChartBuilder()
        self._layout: List[Dict[str, Any]] = []

    def add_chart(
        self,
        fig: Any,
        row: int,
        col: int,
        title: str = None,
    ):
        """Fügt Chart zum Dashboard hinzu."""
        self._layout.append({
            'figure': fig,
            'row': row,
            'col': col,
            'title': title,
        })

    def build(
        self,
        rows: int = 2,
        cols: int = 2,
        height: int = 800,
    ) -> Any:
        """
        Baut Dashboard.

        Args:
            rows: Anzahl Zeilen
            cols: Anzahl Spalten
            height: Gesamthöhe in Pixeln

        Returns:
            Plotly Figure mit Subplots
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Erstelle Subplots
        titles = [item.get('title', '') for item in self._layout]
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=titles[:rows * cols],
        )

        # Füge Charts hinzu
        for item in self._layout:
            chart = item['figure']
            if chart is None:
                continue

            row, col = item['row'], item['col']

            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            title=self.title,
            height=height,
            showlegend=True,
        )

        return fig

    def save(self, output_path: str, format: str = "html") -> str:
        """Speichert Dashboard."""
        dashboard = self.build()
        return self.charts.save(dashboard, output_path, format=format)


# Factory Functions
def get_chart_builder(theme: str = "plotly_white") -> ChartBuilder:
    """Erstellt ChartBuilder."""
    return ChartBuilder(theme=theme)


def get_eda_report() -> EDAReport:
    """Erstellt EDAReport."""
    return EDAReport()


def get_dashboard_builder(title: str = "SCIO Dashboard") -> DashboardBuilder:
    """Erstellt DashboardBuilder."""
    return DashboardBuilder(title=title)
