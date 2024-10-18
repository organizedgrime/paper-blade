use iced::alignment::Vertical;
use iced::Length;
use iced_aw::style::color_picker::Catalog;
use iced_aw::{menu::Item, menu_bar};
use iced_wgpu::Renderer;
use iced_widget::{button, column, container, row, text, Row};
use iced_winit::core::{Color, Element, Theme};
use iced_winit::runtime::{Program, Task};
use strum::IntoEnumIterator;

use crate::render::color::RGBA;
use crate::render::{message::*, state::AppState};

use super::menu::MenuAble;

pub struct Controls {
    pub state: AppState,
}

impl Controls {
    pub fn new() -> Self {
        Self {
            state: AppState::default(),
        }
    }

    pub fn background_color(&self) -> Color {
        self.state.render.background_color
    }
}

impl Controls
where
    Controls: Program,
{
    pub fn button_background<'a>(
        color: RGBA,
    ) -> (impl Fn(&Theme, button::Status) -> button::Style + 'a)
    where
        Theme: button::Catalog,
    {
        move |theme, status| {
            button::text(theme, status).with_background(iced::Background::from(color))
        }
    }
}

impl Program for Controls {
    type Renderer = Renderer;
    type Theme = Theme;
    type Message = PolybladeMessage;

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        // println!("processing!");
        self.state.model.polyhedron.update();
        message.process(&mut self.state)
    }

    fn view(&self) -> Element<Self::Message, Self::Theme, Self::Renderer> {
        let mut button_row = Row::new().spacing(10);
        for (i, color) in self
            .state
            .render
            .picker
            .palette
            .colors
            .iter()
            .cloned()
            .enumerate()
        {
            button_row = button_row.push(
                button("")
                    .style(Self::button_background(color))
                    .width(20)
                    .height(20)
                    .on_press(Self::Message::Render(RenderMessage::ColorPicker(
                        ColorPickerMessage::ChooseColor(i),
                    ))),
            );
        }

        let menu_bar = row![menu_bar!((
            PresetMessage::title(),
            PresetMessage::menu(&())
        )(
            ConwayMessage::title(),
            ConwayMessage::menu(&())
        )(
            RenderMessage::title(),
            RenderMessage::menu(&self.state.render)
        ))]
        .spacing(10.0);

        container(
            column![
                menu_bar.align_y(Vertical::Top),
                button_row,
                iced_widget::Space::new(Length::Fill, Length::Fill),
                button(text(self.state.info.name())).on_press(self.state.info.wiki_message()),
                container(
                    row![
                        column![
                            text("Bowers:"),
                            text("Conway:"),
                            text("Faces:"),
                            text("Edges:"),
                            text("Vertices:"),
                        ],
                        column![
                            text(self.state.info.bowers()),
                            text(&self.state.info.conway),
                            text(self.state.info.faces),
                            text(self.state.info.edges),
                            text(self.state.info.vertices),
                        ]
                    ]
                    .spacing(20)
                    .align_y(Vertical::Bottom)
                )
                .style(iced::widget::container::dark)
            ]
            .spacing(10),
        )
        .padding(10)
        .into()
    }
}
