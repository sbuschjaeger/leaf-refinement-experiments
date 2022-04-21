(Some) documentation is available under http://phynetlab.com/PhyNetLab.pdf
The data is available under http://phynetlab.com/dataset.zip
The VM is available under http://phynetlab.com/vm.zip
    - Unfortunately, the login details are lost for the default user (`student`). I reset the password by mounting the VM, changing the root, resetting the root password and resetting the student password. See also
    
        Mount vmdk images on Linux via `guestmount`: https://stackoverflow.com/questions/22327728/mounting-vmdk-disk-image

        Changing the root filesystems via `chroot`: https://www.makeuseof.com/tag/how-to-reset-any-linux-password/#:~:text=Mount%20the%20linux%20partition%20using,reboot%20to%20restart%20the%20system.
        
        Reset the password via `passwd`: https://www.cyberciti.biz/faq/linux-set-change-password-how-to/
    - There seem to be some older temporary files inside the VM. For compilation I recommend to 
        1) clean 
        2) remove file 