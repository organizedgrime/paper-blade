use crate::wgpu;

// A custom buffer container for dynamic resizing.
pub struct Buffer {
    pub raw: wgpu::Buffer,
    label: &'static str,
    usage: wgpu::BufferUsages,
}

impl Buffer {
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self {
            raw: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            }),
            label,
            usage,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, new_size: u64) {
        self.raw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
    }
}
