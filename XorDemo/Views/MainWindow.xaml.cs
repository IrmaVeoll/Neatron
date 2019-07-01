using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Media;
using OxyPlot;
using OxyPlot.Axes;
using ReactiveUI;
using XorDemo.ViewModels;
using Axis = OxyPlot.Avalonia.Axis;
using LinearAxis = OxyPlot.Avalonia.LinearAxis;

namespace XorDemo.Views
{
    public class MainWindow : ReactiveWindow<MainWindowViewModel>
    {
        public MainWindow()
        {
            void SetupAxis(Axis linearAxis)
            {
                linearAxis.AxislineStyle = LineStyle.Solid;
                linearAxis.AxislineColor = Colors.LightGray;
                linearAxis.TickStyle = TickStyle.Outside;
                linearAxis.TicklineColor = Color.FromRgb(220,220,220);
                linearAxis.MajorTickSize = 6.0;
                linearAxis.MinorTickSize = 4.0;
                linearAxis.MajorGridlineStyle = LineStyle.None;
                linearAxis.MajorGridlineThickness = 1.01;
                linearAxis.MinorGridlineStyle = LineStyle.None;
                linearAxis.MinorGridlineThickness = 1.01;
            }

            InitializeComponent();
            
            SetupAxis(this.FindControl<LinearAxis>("FitY"));
            SetupAxis(this.FindControl<LinearAxis>("FitX"));
            SetupAxis(this.FindControl<LinearAxis>("ComY"));
            SetupAxis(this.FindControl<LinearAxis>("ComX"));
        }

        private void InitializeComponent()
        {
            this.WhenActivated(disposables => { /* Handle view activation etc. */ });
            AvaloniaXamlLoader.Load(this);
        }
    }
}