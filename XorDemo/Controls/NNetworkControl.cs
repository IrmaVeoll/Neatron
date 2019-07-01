using Avalonia;
using Avalonia.Controls.Primitives;
using Avalonia.Media;
using Neat;

namespace XorDemo.Controls
{
    public sealed class NNetworkControl : TemplatedControl
    {
        private NNetworkPresentation _presentation;
        
        public static readonly StyledProperty<double> PenOffsetProperty = 
            AvaloniaProperty.Register<NNetworkControl, double>(nameof(PenOffset));
        
        public static readonly StyledProperty<NetworkTopology> NetworkTopologyProperty = 
            AvaloniaProperty.Register<NNetworkControl, NetworkTopology>(nameof(NetworkTopology));
        
        static NNetworkControl()
        {
            AffectsRender<NNetworkControl>(
                PenOffsetProperty, 
                NetworkTopologyProperty);
            
            AffectsPresentation(
                NetworkTopologyProperty,
                TransformedBoundsProperty);
        }

        public double PenOffset
        {
            get => GetValue(PenOffsetProperty);
            set => SetValue(PenOffsetProperty, value);
        }

        public NetworkTopology NetworkTopology
        {
            get => GetValue(NetworkTopologyProperty);
            set => SetValue(NetworkTopologyProperty, value);
        }

        public override void Render(DrawingContext drawingContext)
        {
            if (_presentation == null)
                UpdatePresentation();
            
            _presentation?.Draw(drawingContext, PenOffset);

//            if (TransformedBounds.HasValue)
//            {
//                drawingContext.DrawGeometry(Brushes.Transparent, new Pen(Brushes.Black), new RectangleGeometry(TransformedBounds.Value.Bounds));
//            }
        }

        private void UpdatePresentation()
        {
            if (!TransformedBounds.HasValue)
                return;
            
            var size = TransformedBounds.Value.Bounds.Size;
            _presentation = NetworkTopology.IsValid && size.Width > 0 && size.Height > 0
                ? new NNetworkPresentation(NetworkTopology, size.Width, size.Height)
                : default;
        }
        
        private static void AffectsPresentation(params AvaloniaProperty[] properties)
        {
            foreach (var property in properties)
                property.Changed.AddClassHandler<NNetworkControl>((c, e) => c.UpdatePresentation());
        }
    }
}
