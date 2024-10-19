use crate::{
    bones::PolyGraph,
    render::{camera::Camera, palette::Palette, polydex::InfoBox},
    Instant,
};

use iced::{time::Duration, Color};
use std::{f32::consts::PI, io::Read as _};
use ultraviolet::Mat4;

use super::{
    message::ColorMethodMessage,
    polydex::{Entry, Polydex},
};

pub struct AppState {
    pub model: ModelState,
    pub render: RenderState,
    pub polydex: Polydex,
    pub info: InfoBox,
}

#[derive(Debug, Clone)]
pub struct RenderState {
    pub camera: Camera,
    pub zoom: f32,
    pub start: Instant,
    pub rotation_duration: Duration,
    pub rotating: bool,
    pub schlegel: bool,
    pub line_thickness: f32,
    pub method: ColorMethodMessage,
    pub picker: ColorPickerState,
    pub background_color: Color,
}

#[derive(Debug, Clone)]
pub struct ColorPickerState {
    pub palette: Palette,
    pub color_index: Option<usize>,
    pub picked_color: Color,
    pub colors: i16,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            zoom: 1.0,
            start: Instant::now(),
            rotation_duration: Duration::from_secs(0),
            rotating: true,
            schlegel: false,
            line_thickness: 2.0,
            method: ColorMethodMessage::Polygon,
            picker: ColorPickerState::default(),
            background_color: Color::WHITE,
        }
    }
}

impl Default for ColorPickerState {
    fn default() -> Self {
        Self {
            palette: Palette::clement(),
            color_index: None,
            picked_color: Color::from_rgba8(0, 0, 0, 1.0),
            colors: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelState {
    pub polyhedron: PolyGraph,
    pub transform: Mat4,
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            polyhedron: PolyGraph::dodecahedron(),
            transform: Mat4::identity(),
        }
    }
}

pub fn load_polydex() -> Result<Polydex, Box<dyn std::error::Error>> {
    let mut polydex = std::fs::File::open("polydex.ron")?;
    let mut polydex_str = String::new();
    polydex.read_to_string(&mut polydex_str)?;
    let polydex: Vec<Entry> = ron::from_str(&polydex_str).map_err(|_| "Ron parsing error")?;
    Ok(polydex)
}

impl Default for AppState {
    fn default() -> Self {
        let info = PolyGraph::default().polydex_entry(&vec![]);
        Self {
            model: ModelState::default(),
            render: RenderState::default(),
            polydex: load_polydex().unwrap_or_default(),
            info,
        }
    }
}

impl AppState {
    pub fn update_state(&mut self, time: Instant) {
        let time = if self.render.rotating {
            time.duration_since(self.render.start)
        } else {
            self.render.rotation_duration
        };

        self.model.polyhedron.update();
        let time = time.as_secs_f32();
        self.model.transform = Mat4::default();
        if self.render.schlegel {
            self.model.transform = Mat4::identity();
        } else {
            self.model.transform = Mat4::from_scale(self.render.zoom)
                * Mat4::from_rotation_x(time / PI)
                * Mat4::from_rotation_y(time / PI * 1.1);
        }
    }
}
