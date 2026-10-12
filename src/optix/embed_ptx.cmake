file(READ "${INPUT}" PTX)
file(WRITE "${OUTPUT}"
	 "#pragma once\nstatic const char xenon_optix_ptx[] = R\"xenon_ptx(${PTX})xenon_ptx\";\n")
