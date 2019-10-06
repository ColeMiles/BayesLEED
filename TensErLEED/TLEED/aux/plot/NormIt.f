C  This program reads theoretical energies in the van Hove format, 
C  normalises them for equal integral intensity, and writes them in a 
C  column format.

      Program NormIt

C  MNBEAMS is maximum number of beams - Note output format to Spec.out MUST
C          be large enough to handle all beams!!
C  MN_ENER is maximum number of energies

      PARAMETER (MNBEAMS=27,MN_ENER=200)

C  NBEAMS is actual number of beams
C  KBEAMS is number of beams as given by NormIt.steu
C  N_NENER is actual number of energies

      INTEGER KBEAMS
      INTEGER NBEAMS,N_ENER

C  Energy0 is original energy, unshifted by inner potential
C  Energy is array of energies encountered
C  Intens is single beam intensities read for each energy
C  Geo is geometry characterisation - currently unused. Geo can be used later
C      to extract a single geometry from an output file containing several
C      different spectra.

      REAL Energy, Energy0, Intens
      DIMENSION Energy(MN_ENER,MNBEAMS),Intens(MN_ENER,MNBEAMS)
      DIMENSION Energy0(MN_ENER,MNBEAMS)

      REAL Geo

C  VvH is const part of inner potential as used in calculation
C  Vrf is const part of inner potential as found by r-factor program
C  VV = Vrf - VvH is shift in inner potential that must be performed here
C  Shift determines whether energy dependent inner potential shift is desired
C  quot is quotient in inner potential of shape V0R=-VV+(E/quot)
C  c1,c2,c3,c4 are coefficients for Rundgren-type inner potential

      Integer Shift
      REAL VV, quot
      REAL c1,c2,c3,c4

C  EMIN is lower boundary of energy range for each beam
C  EMAX is upper boundary of energy range for each beam
C  NDATT contains number of valid data points for each beam in given
C        energy range

      REAL EMIN, EMAX
      DIMENSION EMIN(MNBEAMS),EMAX(MNBEAMS)

      INTEGER NDATT
      DIMENSION NDATT(MNBEAMS)

C  MITTEL contains beam grouping info for each beam
C  CNTBeams counts beams belonging to one beam group
C  GEnergy contains energy values for averaged beams
C  GIntens contains intensity values for averaged beams
C  GNBEAMS is number of beam groups after averaging
C  GNDATT is number of energies in each beam group after averaging


      INTEGER Mittel, CNTBeams
      DIMENSION Mittel(MNBEAMS)

      REAL GEnergy,GIntens
      DIMENSION GEnergy(MN_ENER,MNBEAMS),GIntens(MN_ENER,MNBEAMS)

      INTEGER GNBEAMS,GNDATT
      DIMENSION GNDATT(MNBEAMS)

C  MaxInt is maximum intensity in a beam group

      REAL MaxInt
      DIMENSION MaxInt(MNBEAMS)

C  Norm is to be highest intensity in each beam
C  NormFlag determines whether normalisation is to be performed
C  to factor Norm only or for integral intensities

      REAL Norm
      INTEGER NormFlag

C  Integral is integral intensity for each beam group

      REAL Integral
      DIMENSION Integral(MNBEAMS)

C  Offset is Offset intensity to be added to all spectra

      REAL Offset

C  begin work

      write(6,*) 'Normalisation and column output of calculated',
     +           ' I(E) data.'
      write(6,*) 'reading control input from NormIt.steu'
C  read control input here

      OPEN(7,FILE="NormIt.steu")

C  number of beams

      READ(7,'(I3)') KBEAMS
      write(6,'(I3)') KBEAMS

C  beam grouping

      READ(7,'(40I3)') (Mittel(IBEAM), IBEAM = 1,KBEAMS)
      write(6,'(40I3)') (Mittel(IBEAM), IBEAM = 1,KBEAMS)

C  energy ranges for all beams

      DO 50 IBEAM = 1,KBEAMS

        READ(7,'(2F7.2)') EMIN(IBEAM),EMAX(IBEAM)
        write(6,'(2F7.2)') EMIN(IBEAM),EMAX(IBEAM)

 50   CONTINUE

C  inner potential from calculation

      READ(7,*) VvH
      write(6,*) VvH

C  inner potential from r-factor subroutine

      READ(7,*) Vrf
      write(6,*) Vrf

      VV = Vrf - VvH

C  do energy-dependent shift E = E - VV + E/quot ?
C  do Rundgren-type energy-dependent shift ?

      READ(7,'(I3)') Shift
      write(6,'(I3)') Shift

C  read quot for shift formula

      if (Shift.eq.1) then

        READ(7,'(F7.2)') quot
        write(6,*) "Linear inner potential shift used: E/", quot

      else if (Shift.eq.2) then

        read(7,*) c1,c2,c3,c4
        write(6,*) "Using Rundgren-type energy-dependent inner ",
     +             "potential max( c1 , (c2+c3/sqrt(E+c4)) )with"
        write(6,*) "c1 = ",c1
        write(6,*) "c2 = ",c2
        write(6,*) "c3 = ",c3
        write(6,*) "c4 = ",c4

      else

        read(7,*)
        write(6,*) "No energy-dependent rescaling of energy axis."

      end if     

C  read normalisation factor for highest peak

      READ(7,'(F7.2)') Norm
      write(6,'(F7.2)') Norm

C  do integral normalisation ?

      READ(7,'(I3)') NormFlag
      write(6,'(I3)') NormFlag

C  Offset for intensities after treatment

      READ(7,'(F7.2)') Offset
      write(6,'(F7.2)') Offset


      write(6,*) 'finished reading NormIt.steu'

C  read input spectra now.....

C  van Hove file is expected at stdin!!

C  read title and beam name

      READ(5,*)
      READ(5,*) NBEAMS

      IF (KBEAMS.ne.NBEAMS) THEN

        write(6,*) 'Beam number in NormIt.steu wrong!'
        STOP

      END IF

      IF (NBEAMS.gt.MNBEAMS) THEN

        write(6,*) 'Too many beams for available dimensions 
     +- change these first!!'
        STOP

      END IF

      DO 100 IBEAM = 1,NBEAMS

        READ(5,*)

 100  CONTINUE

C  Start reading of energy / intensity information here (Loop 200)
C  read only energies for first beam so far

      I_ENER = 0

 200  CONTINUE

        I_ENER = I_ENER + 1


        READ(5,'(F7.2,F7.4,4E14.5,5(/5E14.5))',ERR=250,END=300)
     +      Energy0(I_ENER,1),Geo,(Intens(I_ENER,IBEAM), IBEAM=1,NBEAMS)

C  Check consistency of values

        IF ((Energy0(I_ENER,1).eq.Energy0(I_ENER-1,1)).or.
     +      (Energy0(I_ENER,1).eq.0.)) THEN

          write(6,*) 'Spectrum contains more than one geometry 
     +or illegal energy values: Energy no. ', I_ENER
          STOP

        END IF

        IF (I_ENER.gt.MN_ENER) THEN

          write(6,*) 'Insufficient Dimensions for energy 
     +- change dimensions first!'
          STOP

        END IF

C  do inner potential shift of read energy values here

        IF (Shift.eq.1) THEN

          Energy(I_ENER,1)=Energy0(I_ENER,1) - VV +
     +                      Energy0(I_ENER,1)/quot

        ELSE IF (Shift.eq.2) THEN

          Energy(I_ENER,1)=Energy0(I_ENER,1) - VV +
     +    max(c1,c2-c3/sqrt(Energy0(I_ENER,1)+c4))

        ELSE

          Energy(I_ENER,1)=Energy0(I_ENER,1) - VV

        END IF

      GO TO 200

C  In case of error while reading

 250  CONTINUE
    
        write(6,*) 'Illegal values read, Energy no.',I_ENER
        STOP

C  End readin loop

 300  CONTINUE

      N_ENER = I_ENER - 1

      write(6,*) 'reading of exp data ok'

C  expand read energies for all beams now

      DO 350 I_ENER = 1,N_ENER

        DO 370 IBEAM = 1,NBEAMS

          Energy(I_ENER,IBEAM)=Energy(I_ENER,1)

 370    CONTINUE

 350  CONTINUE

C  now restrict beams to given energy ranges

      DO 400 IBEAM = 1,NBEAMS,1

        IDATT = 0

        DO 500 I_ENER = 1,N_ENER

          IF(Energy(I_ENER,IBEAM).lt.EMIN(IBEAM)) GO TO 500

          IF(Energy(I_ENER,IBEAM).gt.EMAX(IBEAM)) GO TO 500

            IDATT = IDATT + 1

            Energy(IDATT,IBEAM) = Energy(I_ENER,IBEAM)
 
            Intens(IDATT,IBEAM) = Intens(I_ENER,IBEAM)

 500    CONTINUE

        NDATT(IBEAM) = IDATT

 400  CONTINUE

      write(6,*) 'energy ranges ok'      

C  now average beam groups together, beam group by beam group
C  store new averaged values in GEnergy, GIntens.
C  ONLY THESE VALUES WILL BE USED IN THE FOLLOWING PARTS!

      GNBEAMS = 0

      DO 600 IGroup = 1,NBEAMS,1

        CNTBeams = 0
        GNDATT(IGroup) = 0

        DO 700 IBEAM = 1,NBEAMS,1

          IF (Mittel(IBEAM).eq.IGroup) THEN

            CNTBeams = CNTBeams + 1
            GNDATT(IGroup) = NDATT(IBEAM)
            
            IF (CNTBeams.eq.1) THEN

              GNBEAMS = GNBEAMS + 1

            END IF

            DO 800 IDATT = 1,NDATT(IBEAM),1

              IF (CNTBeams.eq.1) THEN

                GIntens(IDATT,IGroup) = Intens(IDATT,IBEAM)
                GEnergy(IDATT,IGroup) = Energy(IDATT,IBEAM)

              ELSE

                GIntens(IDATT,IGroup) = GIntens(IDATT,IGroup) +
     +                                  Intens(IDATT,IBEAM)

              END IF

 800        CONTINUE

          END IF

 700    CONTINUE

C  finish averaging over intensities now
C  and save maximum intensity in that beam

        MaxInt(IGroup) = 0.

        DO 900 IDATT = 1,GNDATT(IGroup)

          GIntens(IDATT,IGroup) = GIntens(IDATT,IGroup) /
     +                              REAL(CNTBeams)

          IF (GIntens(IDATT,IGroup).gt.MaxInt(IGroup)) THEN

            MaxInt(IGroup) = GIntens(IDATT,IGroup)

          END IF

 900    CONTINUE

 600  CONTINUE

      write(6,*) 'averaging ok'
      write(6,*) 'normalisation constants:'

C  From here on, only GEnergy, GIntens, GNBEAMS, GNDATT should be used

C  Now normalise spectra in the desired fashion
C  Make sure that beam contains data points at all!

      DO 1000 IGroup = 1,GNBEAMS

        write(6,*) IGroup, MaxInt(IGroup)

        DO 1100 IDATT = 1,GNDATT(IGroup)

          IF (MaxInt(IGroup).gt.1.E-07) THEN

            GIntens(IDATT,IGroup) = GIntens(IDATT,IGroup) * 
     +                              Norm / MaxInt(IGroup)
          END IF

 1100   CONTINUE

 1000 CONTINUE

      write(6,*) 'Normalisation ok'

C  If normalisation to average integral value is desired, do that now

      IF (NormFlag.eq.1) THEN

        DO 1200 IGroup = 1,GNBEAMS

          Integral(IGroup) = 0.

          DO 1300 IDATT = 1,GNDATT(IGroup)

            Integral(IGroup) = Integral(IGroup) + GIntens(IDATT,IGroup)

 1300     CONTINUE

          Integral(IGroup) = Integral(IGroup)/GNDATT(IGroup)

 1200   CONTINUE

        write(6,*) 'Integral intensity for beam 1:',
     +              Integral(2)

        DO 1400 IGroup = 1,GNBEAMS

          DO 1500 IDATT = 1,GNDATT(IGroup)

            GIntens(IDATT,IGroup) = GIntens(IDATT,IGroup) * 
     +                              Integral(2) / Integral(IGroup)

 1500     CONTINUE

 1400   CONTINUE

      END IF

C  add offset intensity to spectra

      DO 1600 IGroup = 1,GNBEAMS

        DO 1700 IDATT = 1, GNDATT(IGroup)

          GIntens(IDATT,IGroup) = GIntens(IDATT,IGroup) + Offset

 1700   CONTINUE

 1600 CONTINUE

C  write zeros to all unused parts of energy / intensity arrays
C  to prepare for wacko f77 output standard 

      DO 1800 IGroup = 1,GNBEAMS

        DO 1900 IDATT = GNDATT(IGroup)+1, N_ENER

          GEnergy(IDATT,IGroup) = 0.
          GIntens(IDATT,IGroup) = 0.

 1900   CONTINUE

 1800 CONTINUE

C  Alas, it is done! Now write the results to file Spec.out in column format

      OPEN(8,FILE='Spec.out')

      DO 2000 IDATT = 1,N_ENER

        write(8,'(80F8.2)') 
     + (GEnergy(IDATT,IGroup),GIntens(IDATT,IGroup),IGroup=1,GNBEAMS)

 2000 CONTINUE

C  That's all folks

      END
