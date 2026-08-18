{
  description = "Delivery";
  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/develop";
    # need to use develop since ros-gz-bridge is not yet available on master
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs = { self, nix-ros-overlay, nixpkgs, nixpkgs-unstable }:
    nix-ros-overlay.inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ 
            nix-ros-overlay.overlays.default
          ];
        };

        unstablePkgs = import nixpkgs-unstable {
          inherit system;
          config.allowUnfree = true;
        };

        # Pull the specific Python interpreter ROS Humble is using
        pythonWithPackages = pkgs.rosPackages.${rosDistro}.python.withPackages (p: with p; [
          numpy
          scipy
          opencv4
          debugpy
          osmnx
          shapely
          jax
          optax
        ]);

        rosDistro = "humble";
      in {
        devShells.default = pkgs.mkShell {
          name = "Delivery";
          packages = with pkgs; [
            colcon
            opencv
            pythonWithPackages
            unstablePkgs.foxglove-studio

            (with rosPackages.${rosDistro}; buildEnv {
              paths = [
                ament-cmake
                ament-cmake-core
                ament-cmake-python
                python-cmake-module
                ros-core
                rclcpp
                rclpy
                rviz2
                cv-bridge
                joy
                joy-linux
                joy-teleop
                ros-gz-bridge
                ros-gz-interfaces
                xacro
                teleop-twist-keyboard
                robot-state-publisher
                foxglove-bridge
                ros-ign-bridge
                tf2-tools
              ];
            })
          ];
        };
      });
  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" ];
    extra-trusted-public-keys = [ "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo=" ];
    permittedInsecurePackages = [
      "freeimage-3.18.0-unstable-2024-04-18"
    ];
  };
}
