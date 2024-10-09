use iced::alignment::{Horizontal, Vertical};
use iced::Length;
use iced_aw::style::color_picker::Catalog;
use iced_aw::{color_picker, menu::Item, menu_bar};
use iced_aw::{menu, menu_items, Menu};
use iced_wgpu::Renderer;
use iced_widget::{button, column, container, row, shader, slider, text, Button, Row};
use iced_winit::core::{Color, Element, Length::*, Theme};
use iced_winit::runtime::{Program, Task};
use strum::IntoEnumIterator;

use crate::render::{menu::ColorPickerBox, message::*, state::AppState};

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

impl Program for Controls {
    type Renderer = Renderer;
    type Theme = Theme;
    type Message = PolybladeMessage;

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        println!("processing!");
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
                    .style(move |theme, status| {
                        button::text(theme, status).with_background(iced::Background::from(color))
                    })
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
        /* container(
            column![
                //       button_row,
                // Actual shader of the program
                // container(shader(self.state).width(Length::Fill).height(Length::Fill)),
                // Info
                column![
                    column![
                        button(text(self.state.info.name()))
                            .on_press(self.state.info.wiki_message()),
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
                    ],
                    row![
                        text("Colors: "),
                        text(self.state.render.picker.colors.to_string()),
                        slider(
                            1..=self.state.render.picker.palette.colors.len() as i16,
                            self.state.render.picker.colors,
                            |x| PolybladeMessage::Render(RenderMessage::ColorPicker(
                                ColorPickerMessage::ChangeNumber(x)
                            ))
                        )
                        .step(1i16)
                    ],
                    row![
                        text("Size: "),
                        text(self.state.model.scale.to_string()),
                        slider(0.0..=10.0, self.state.model.scale, |v| {
                            PolybladeMessage::Model(ModelMessage::ScaleChanged(v))
                        })
                        .step(0.1)
                    ],
                ]
            ]
            .spacing(10), // .push(cp),
        ) */
        // .width(Length::Fill)
        // .height(Length::Fill)
        // .align_x(Horizontal::Center)
        // .align_y(Vertical::Center)
        // .padding(10);

        container(
            column![
                menu_bar.align_y(Vertical::Top),
                button(text(self.state.info.name())).on_press(self.state.info.wiki_message()),
                row![
                    button_row,
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
            ]
            .spacing(10),
        )
        .padding(10)
        .into()
    }
}
