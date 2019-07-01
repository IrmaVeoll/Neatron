using System;
using System.Collections.Generic;
using System.Linq;
using Avalonia;
using Avalonia.Media;
using Avalonia.Visuals.Platform;
using Neat;
using Brushes = Avalonia.Media.Brushes;
using Color = Avalonia.Media.Color;
using Pen = Avalonia.Media.Pen;

namespace XorDemo.Controls
{
    internal class NNetworkPresentation
    {
        private const double Spread = Math.PI / 4;
        private const double Intense = 230;
        private const double NeuronRadius = 15.0;
        private const double SensorRadius = 10.0;
        private const double EffectorRadius = 20.0;
        
        private static readonly Pen NeuronPen = new Pen(new SolidColorBrush(Color.FromRgb(250, 130, 130)), 2);
        private static readonly Color NeuronColor = Color.FromRgb(245, 192, 192);
        
        private static readonly Pen BiasPen = new Pen(new SolidColorBrush(Color.FromRgb(125, 182, 208)), 1.5);
        private static readonly Brush BiasBrush = new SolidColorBrush(Color.FromRgb(209,233,245));
        
        private static readonly Pen SensorPen = new Pen(new SolidColorBrush(Color.FromRgb(112, 198, 112)), 1.5);
        private static readonly Brush SensorBrush = new SolidColorBrush(Color.FromRgb(230, 240, 230));
        
        private static readonly Pen EffectorPen = new Pen(new SolidColorBrush(Color.FromRgb(128, 128, 255)), 2);
        private static readonly Color EffectorColor = Color.FromRgb(192, 192, 245);

        private const double Asymmetry = 0.6;
        private readonly Dictionary<int, List<((int, double), Point)>> _inputSlots;
        private readonly Dictionary<int, List<((int, double), Point)>> _outputSlots;

        private readonly Dictionary<int, (Point, double, Pen, Brush, NeuronForm)> _neurons =
            new Dictionary<int, (Point, double, Pen, Brush, NeuronForm)>();
        
        private readonly double _intense;
        private readonly double _neuronRadius;
        private readonly double _sensorRadius;
        private readonly double _effectorRadius;
        private readonly double _scale;

        private enum NeuronForm
        {
            Circle,
            Rectangle
        }

        public NNetworkPresentation(NetworkTopology nNetwork, double width, double height)
        {
            _scale = Math.Min(width, height) / 800;
            
            _intense = Intense * _scale;
            _neuronRadius = NeuronRadius * _scale;
            _sensorRadius = SensorRadius * _scale;
            _effectorRadius = EffectorRadius * _scale;
            
            (double, Pen, Brush) GetDrawingParameters(Point point, int layer, int layersCount, int neuronNum)
            {
                double size;
                Brush brush;
                Pen pen;
                
                if (layer == 0 && neuronNum == 0)
                {
                    size = _sensorRadius;
                    brush = BiasBrush;
                    pen = BiasPen;
                }
                else if (layer == 0)
                {
                    size = _sensorRadius;
                    brush = SensorBrush;
                    pen = SensorPen;
                }
                else if (layer < layersCount - 1)
                {
                    size = _neuronRadius;
                    var gradientBrush = new RadialGradientBrush()
                    {
                        Radius = 1,
                        Center = new RelativePoint(point, RelativeUnit.Absolute)
                    };
                    gradientBrush.GradientStops.Add(new GradientStop(Colors.White, 0));
                    gradientBrush.GradientStops.Add(new GradientStop(NeuronColor, _neuronRadius));
                    brush = gradientBrush;
                    pen = NeuronPen;
                }
                else
                {
                    size = _effectorRadius;
                    var gradientBrush = new RadialGradientBrush()
                    {
                        Radius = 1,
                        Center = new RelativePoint(point, RelativeUnit.Absolute)
                    };
                    gradientBrush.GradientStops.Add(new GradientStop(Colors.White, 0));
                    gradientBrush.GradientStops.Add(new GradientStop(EffectorColor, _effectorRadius));
                    brush = gradientBrush;
                    pen = EffectorPen;
                }

                return (size, pen, brush);
            }
            
            var layers = nNetwork.LayerRanges;

            const double aspect = 3f / 2.8f;
            var cx = Math.Min(width, height * aspect);
            var cy = Math.Min(height, width / aspect);
            for (var layer = 0; layer < layers.Count; layer++)
            {
                var neuronOnLayerCount = layers[layer].Item2;
                var distance = cy / neuronOnLayerCount;
                var horizontalPosition = width / 2 - cx / 2 + layer * cx / (layers.Count - 1);
                var range = layers[layer];
                for (var neuronNum = 0; neuronNum < layers[layer].Item2; neuronNum++)
                {
                    var verticalPosition =
                        height / 2 - cy / 2 + neuronNum * distance + distance / 2;
                    var point = new Point(horizontalPosition, verticalPosition);
                    var (size, pen, brush) = GetDrawingParameters(point, layer, layers.Count, neuronNum);
                    _neurons[range.Item1 + neuronNum] =
                        (point, size, pen, brush,
                            layer == 0 && neuronNum == 0 ? NeuronForm.Rectangle : NeuronForm.Circle);
                }
            }
    
            var (neuronInputsWithLinks, neuronOutputsWithLinks) = BuildLinkLists(nNetwork);
            _outputSlots = BuildSlots(neuronOutputsWithLinks, false);
            _inputSlots = BuildSlots(neuronInputsWithLinks, true);
        }

        private Dictionary<int, List<((int, double), Point)>> 
            BuildSlots(Dictionary<int, List<(int, double)>> neuronOutputsWithLinks, bool isInput)
        {
            double Distance(Point p1, Point p2)
            {
                return Math.Sqrt((p2.X - p1.X) * (p2.X - p1.X) + (p2.Y - p1.Y) * (p2.Y - p1.Y));
            }
            var intense = isInput ? _intense / Asymmetry : _intense * Asymmetry;
            return neuronOutputsWithLinks.ToDictionary(
                kv => kv.Key,
                kv =>
                {
                    var neuronNumber = kv.Key;
                    var neuronPoint = _neurons[neuronNumber];
                    kv.Value.Sort((n1, n2) =>
                    {
                        if (n1.Item1 == neuronNumber) return 1;
                        if (n2.Item1 == neuronNumber) return -1;
                        var p1 = _neurons[n1.Item1].Item1;
                        var p2 = _neurons[n2.Item1].Item1;
                        var s1 = (p1.Y - neuronPoint.Item1.Y) / Distance(p1, neuronPoint.Item1);
                        var s2 = (p2.Y - neuronPoint.Item1.Y) / Distance(p2, neuronPoint.Item1);
                        return Math.Sign(s2 - s1);
                    });
                    var angleStep = Spread / kv.Value.Count;
                    return kv.Value.Select((n, i) =>
                    {
                        var angle = i * angleStep - Spread / 2 + angleStep / 2;
                        var anglePoint = new Point(_intense * Math.Cos(angle) * (isInput ? -1 : 1) +  neuronPoint.Item1.X,
                            -intense * Math.Sin(angle) + neuronPoint.Item1.Y);
                        return (n, anglePoint);
                    }).ToList();
                });
        }

        private static (Dictionary<int, List<(int, double)>>, Dictionary<int, List<(int, double)>>) BuildLinkLists(NetworkTopology nNetwork)
        {
            var neuronInputs = new Dictionary<int, List<(int, double)>>();
            var neuronOutputs = new Dictionary<int, List<(int, double)>>();

            void AddSlot(IDictionary<int, List<(int, double)>> slotsDictionary, int from, int to, double weight)
            {
                if (!slotsDictionary.TryGetValue(from, out var slots))
                {
                    slots = new List<(int, double)>();
                    slotsDictionary[from] = slots;
                }
                slots.Add((to, weight));
            }

            for (var neuronIdx = 0; neuronIdx < nNetwork.Links.Count; ++neuronIdx)
            {
                var links = nNetwork.Links[neuronIdx];
                for (var linkIdx = 0; linkIdx < links.Count; ++linkIdx)
                {
                    var link = links[linkIdx];
                    AddSlot(neuronOutputs, neuronIdx, link.TargetIdx, link.Weight);
                    AddSlot(neuronInputs, link.TargetIdx, neuronIdx, link.Weight);
                }
            }

            return (neuronInputs, neuronOutputs);
        }

        private static void DrawEllipse(DrawingContext drawingContext, Point point, double radius, Pen pen, IBrush brush)
        {
            var geometry = new EllipseGeometry(new Rect(
                new Point(point.X - radius,
                point.Y - radius),
                new Point(point.X + radius,
                point.Y + radius)));
            drawingContext.DrawGeometry(brush, pen, geometry);
        }

        private static void DrawRectangle(DrawingContext drawingContext, Point point, double size, Pen pen, IBrush brush)
        {
            var geometry = new RectangleGeometry(new Rect(
                new Point(point.X - size,
                    point.Y - size),
                new Point(point.X + size,
                    point.Y + size)));
            drawingContext.DrawGeometry(brush, pen, geometry);
        }

        public void Draw(DrawingContext drawingContext, double penOffset)
        {
            var neuronPen = new Pen(Brushes.Black);

            void DrawBezier(Point p1, Point p2, Point p3, Point p4, double weight)
            {
                if (weight == 0)
                    return;

                var width = (.5 + Math.Log(1 + Math.Abs(weight))) * _scale;

                var path = new PathGeometry();
                using (var context = new PathGeometryContext(path))
                {
                    context.BeginFigure(p1, false);
                    context.CubicBezierTo(p2, p3, p4);
                    context.EndFigure(false);

                    var style = new DashStyle(new[] { 5d / width, 2 / width }, penOffset);
                    var linkPen = new Pen(weight < 0 ? Brushes.Blue : Brushes.Red, width, style);

                    drawingContext.DrawGeometry(null, linkPen, path);
                }
            }

            foreach (var item in _outputSlots)
            {
                var sourceNeuron = item.Key;
                var sourcePoint = _neurons[sourceNeuron];
                var slots = item.Value;
                foreach (var slot in slots)
                {
                    var (targetNeuron, outputAnglePoint) = slot;
                    var (_, inputAnglePoint) = _inputSlots[targetNeuron.Item1].Single(i => i.Item1.Item1 == sourceNeuron);
                    var targetPoint = _neurons[targetNeuron.Item1];
                    DrawBezier(sourcePoint.Item1, outputAnglePoint, inputAnglePoint, targetPoint.Item1, targetNeuron.Item2);
                }
            }
            
            foreach (var neuronPoint in _neurons.Select(item => item.Value))
            {
                if (neuronPoint.Item5 == NeuronForm.Circle)
                    DrawEllipse(drawingContext, neuronPoint.Item1, neuronPoint.Item2, neuronPoint.Item3, neuronPoint.Item4);
                else
                    DrawRectangle(drawingContext, neuronPoint.Item1, neuronPoint.Item2, neuronPoint.Item3, neuronPoint.Item4);
            }            
        }
    }
}
