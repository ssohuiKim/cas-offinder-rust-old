use opencl3::*;
use std::thread;
pub struct OclRunConfig {
    devices: Vec<(platform::Platform, Vec<device::Device>)>,
}

pub enum OclDeviceType {
    CPU,
    GPU,
    ACCEL,
    ALL,
}

fn to_system_type(ty: OclDeviceType) -> u64 {
    match ty {
        OclDeviceType::CPU => device::CL_DEVICE_TYPE_CPU,
        OclDeviceType::GPU => device::CL_DEVICE_TYPE_GPU,
        OclDeviceType::ACCEL => device::CL_DEVICE_TYPE_ACCELERATOR,
        OclDeviceType::ALL => device::CL_DEVICE_TYPE_ALL,
    }
}

pub fn get_avali_ocl_devices(
    ty: OclDeviceType,
) -> Result<Vec<(platform::Platform, Vec<device::Device>)>> {
    let mut devices: Vec<(platform::Platform, Vec<device::Device>)> = Vec::new();
    let sys_ty = to_system_type(ty);
    let platforms = platform::get_platforms()?;
    for plat in platforms {
        devices.push((
            plat,
            plat.get_devices(sys_ty)?
                .iter()
                .copied()
                .map(device::Device::new)
                .collect(),
        ));
    }
    Ok(devices)
}

impl OclRunConfig {
    pub fn new(ty: OclDeviceType) -> Result<OclRunConfig> {
        Ok(OclRunConfig {
            devices: get_avali_ocl_devices(ty)?,
        })
    }
    pub fn is_empty(&self) -> bool {
        self.devices.iter().all(|(_, devs)| devs.is_empty())
    }
    pub fn get_device_strs(&self) -> Vec<String> {
        if self.is_empty() {
            let n_threads: usize = thread::available_parallelism().unwrap().into();
            vec![format!(
                "Rust CPU implementation using {} threads",
                n_threads
            )]
        } else {
            self.devices
                .iter()
                .flat_map(|(_, devs)| devs.iter())
                .map(|d| d.name().unwrap())
                .collect()
        }
    }
    pub fn get(&self) -> &Vec<(platform::Platform, Vec<device::Device>)> {
        &self.devices
    }
}
