This tarball contains the chi^2 maps in 2D ($\Delta m^2_32$ and $\sin^2(\theta_{23})$) that corresponds to the data in https://arxiv.org/abs/1707.07081

Apart from the README, there are 8 other files:

-The three with NH in the title contain the results assuming Normal Hierarchy, while the three with IH contain the results assuming Inverted Hierarchy.  Each set of three files hold the same information, just in different formats.

The .dat file contains the value of $\Delta m^2_32$ in eV^2 (labeled "dm32"), the value of $\sin^2(\theta_{23})$ (labeled "sin^2(theta23)"), and the corresponding $\Delta \chi^2$ (labeled "DeltaChi2").

The .pckl files contain a dictionary with keys "dm32", "sin2_theta23", and "delta_chi2".

The .root files contain a tree named chi2_map with branches the same name as the .pckl files.

-The file named contours.png is an example of the 90% CL contour you should get from this data release (dark green and labeled "IC2017 without FC [NH]") compared to that in the arxiv paper (burnt orange and labeled "IC2017 FC [NH]").  The different between these two curves is the dark green one is drawn using a constant delta chi^2 value of 4.6 while the burnt orange was derived using the Feldom-Cousins approach.

-IC2017_90CL_FC.dat is the 90% CL contours obtained using the Feldom-Cousins approach.  The first column has the values of sin^2(\theta_23) while the second column contain the values of $\Delta m^2_32$ in units of 10^-3 eV^2.



A few notes on this data release:

-The chi2 maps are only done in 2D, $\Delta m^2_32$ and $\sin^2(\theta_{23})$.  This is because IceCube is relatively insensitive to the effects of both $theta_{13}$ and $delta_{CP}$, who only change the oscillation probabilities at low energies.  Before fitting we tested the effect of both $\theta_{13}$ and $\delta_{CP}.  Including $\theta_{13}$ changed the size of our contours in $\Delta m^2_32$ and $\sin^2(\theta_{23})$ by less than 0.1%.  There was no impact on the contours from $\delta_{CP}$.  For global fits, this extra space can be considered flat.

-The files have the $Delta \chi^2$ calculated with the local minimum for that hierarchy, not the global minimum between the two hierarchies.  

