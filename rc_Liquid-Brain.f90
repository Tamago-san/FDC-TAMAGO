!-----------------------------------------------------------------------------
!FDC
!-----------------------------------------------------------------------------
module cylinder
    integer(kind=4) :: OUT_NODE = 0
    integer(kind=4) :: IN_NODE = 0
    integer(kind=4) :: TRANING_STEP = 0
    integer(kind=4) :: RC_STEP = 0
    real(kind=8) :: TYU =0.d0 !信号の平行移動(1なら中心１)
    real(kind=8) :: TYO =1.2d0 !信号の大きさ（１なら大きさ１）
    real(kind=8) :: TYK =5.d0 !5なら信号の大きさ1に（5をデフォルトとする）（-0.5～+0.5）
    integer(kind=4), parameter :: NXmin=-20,NXmax=200 ! 計算領域の形状
    integer(kind=4), parameter :: NYmin=-50,NYmax=50  ! 計算領域の形状
    real(kind=8), parameter :: Xmax=20.d0             ! 計算領域の大きさ
    real(kind=8), parameter :: D=2.d0                 ! 円柱の直径
    real(kind=8), parameter :: U=2.d0                 ! 流入速度
    real(kind=8), parameter :: NU=0.04d0              ! 動粘性係数
    real(kind=8), parameter :: dt=0.01d0              ! 時間刻み
    integer(kind=4), parameter :: Nstep=200000         ! LYトータルステップ数
    integer(kind=4), parameter :: iout=100           ! 出力間隔
    integer(kind=4), parameter :: iout_display=500    ! コンパイル画面の出力間隔
    integer(kind=4), parameter :: iSep=2              ! 出力データの間引き
    real(kind=8) :: Xmin,dX
    real(kind=8) :: Ymax,Ymin
    integer(kind=4) :: iX1(NYmin:NYmax),iX2(NYmin:NYmax)
    integer(kind=4) :: iY1(NXmin:NXmax),iY2(NXmin:NXmax)
    integer(kind=4) :: iXb1,iXb2
    integer(kind=4) :: istep
    integer(kind=4), parameter :: Ly_skip_step = 15000 !撹乱が定常、周期まで25000ステップ（250タイム）
    integer(kind=4), parameter :: skip_step = 20000 !定常，周期まで(200time)
    integer(kind=4), parameter :: Ly_out = 100 !リアプノフ指数の測定ステップ数500ステップ
    real(kind=8) :: abs_dV0
    real(kind=8) :: Vx_tmp(NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: Vy_tmp(NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: Re_tmp_dble
    real(kind=8) :: U_tmp
    character(6) :: Re_tmp_name
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !-------------------------------------------------------
    integer,parameter :: tate_y = 20
    integer,parameter :: yoko_x = 100
    integer,parameter :: RC_NODE = tate_y * yoko_x
    integer,parameter :: heikouidou_x = 11
    integer,parameter :: heikouidou_y = -5
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    contains
!-----------------------------------------------------------------------------
    subroutine cal_rhs(Vx,Vy,P,Rx,Ry)
!-----------------------------------------------------------------------------
! ■ 運動方程式の右辺の計算
!-----------------------------------------------------------------------------
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: C1,C2
        C1=1.d0/(2.d0*dX)
        C2=1.d0/(dX**2)
        Rx(:,:)=0.d0
        Ry(:,:)=0.d0
        do iX=NXmin+1,NXmax-1
        do iY=NYmin+1,NYmax-1
            Rx(iX,iY)=-Vx(iX,iY)*C1*(Vx(iX+1,iY)-Vx(iX-1,iY))                 &
                        -Vy(iX,iY)*C1*(Vx(iX,iY+1)-Vx(iX,iY-1))                 &
                        +NU       *C2*(Vx(iX+1,iY)-2.d0*Vx(iX,iY)+Vx(iX-1,iY))  &
                        +NU       *C2*(Vx(iX,iY+1)-2.d0*Vx(iX,iY)+Vx(iX,iY-1))  &
                        -          C1*(P(iX+1,iY)-P(iX-1,iY))
            Ry(iX,iY)=-Vx(iX,iY)*C1*(Vy(iX+1,iY)-Vy(iX-1,iY))                 &
                        -Vy(iX,iY)*C1*(Vy(iX,iY+1)-Vy(iX,iY-1))                 &
                        +NU       *C2*(Vy(iX+1,iY)-2.d0*Vy(iX,iY)+Vy(iX-1,iY))  &
                        +NU       *C2*(Vy(iX,iY+1)-2.d0*Vy(iX,iY)+Vy(iX,iY-1))  &
                        -          C1*(P(iX,iY+1)-P(iX,iY-1))
        enddo
        enddo
        end subroutine cal_rhs
!-----------------------------------------------------------------------------
    subroutine ibm(Vx,Vy,Rx,Ry)
!-----------------------------------------------------------------------------
! ■ 埋め込み境界法
!-----------------------------------------------------------------------------
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
        do iX=iXb1,iXb2
        do iY=iY1(iX),iY2(iX)
            Rx(iX,iY)=Rx(iX,iY)-Vx(iX,iY)/dt
            Ry(iX,iY)=Ry(iX,iY)-Vy(iX,iY)/dt
        enddo
        enddo
        end subroutine ibm
!-----------------------------------------------------------------------------
    subroutine RC_ibm(Vx,Vy,Rx,Ry,U_ibm,O_O)
!-----------------------------------------------------------------------------
! ■ 埋め込み境界法
!-----------------------------------------------------------------------------
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: U_ibm(:,:)
        real(kind=8) :: R
        real(kind=8) :: sin1,cos1
        integer :: step,O_O
      
      
        select case(O_O)
            case (1)
                do iX=iXb1,iXb2
                do iY=iY1(iX),iY2(iX)
                    X=dX*real(iX,8)
                    Y=dX*real(iY,8)
                    R=sqrt(X**2+Y**2)
                    sin1 =Y / R
                    cos1 =X / R
    !                write(*,*) -sin1 *U_tr(istep,IN_NODE)*R
                    Rx(iX,iY)=Rx(iX,iY)+(-Y *U_ibm(istep,IN_NODE) - Vx(iX,iY))/dt
                    Ry(iX,iY)=Ry(iX,iY)+( X *U_ibm(istep,IN_NODE) - Vy(iX,iY))/dt
                enddo
                enddo
            case (2)
                do iX=iXb1,iXb2
                do iY=iY1(iX),iY2(iX)
                    X=dX*real(iX,8)
                    Y=dX*real(iY,8)
                    R=sqrt(X**2+Y**2)
                    sin1 =Y / R
                    cos1 =X / R
                    Rx(iX,iY)=Rx(iX,iY)+(-Y *U_ibm(istep,IN_NODE) - Vx(iX,iY))/dt
                    Ry(iX,iY)=Ry(iX,iY)+( X *U_ibm(istep,IN_NODE) - Vy(iX,iY))/dt
                enddo
                enddo
        end select
        end subroutine RC_ibm
        
    subroutine march(Vx,Vy,P,U_mar,O_O)
!-----------------------------------------------------------------------------
! ■ 時間発展
!-----------------------------------------------------------------------------
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: U_mar(:,:)
        integer O_O
        
        call cal_rhs(Vx,Vy,P,Rx,Ry)
        select case(O_O)
            case (0)
                call ibm(Vx,Vy,Rx,Ry)
            case (1)
                call RC_ibm(Vx,Vy,Rx,Ry,U_mar,1)
            case (2)
                call RC_ibm(Vx,Vy,Rx,Ry,U_mar,2)
        end select
    
        Vx(:,:)=Vx(:,:)+Rx(:,:)*dt
        Vy(:,:)=Vy(:,:)+Ry(:,:)*dt
        call poisson(Vx,Vy,P)
    end subroutine march
 !-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
    subroutine poisson(Vx,Vy,P)
!-----------------------------------------------------------------------------
! ■ 圧力についての Poisson 方程式
!-----------------------------------------------------------------------------
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: P2(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: NLx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: NLy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: RHS(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: C1,C2,C13,CRedX
        real(kind=8) :: alpha
        real(kind=8) :: delta
        real(kind=8) :: divU,divN
        integer(kind=4), parameter :: itrmax=10000
        real(kind=8), parameter :: eps=1.d-2
        C1=1.d0/(2.d0*dX)
        C2=dX**2
        C13=1.d0/3.d0
        CRedX=2.d0/(Re*dX)
        alpha=0.5d0
        do iX=NXmin+1,NXmax-1
        do iY=NYmin+1,NYmax-1
            NLx(iX,iY)=Vx(iX,iY)*C1*(Vx(iX+1,iY)-Vx(iX-1,iY)) &
                      +Vy(iX,iY)*C1*(Vx(iX,iY+1)-Vx(iX,iY-1))
            NLy(iX,iY)=Vx(iX,iY)*C1*(Vy(iX+1,iY)-Vy(iX-1,iY)) &
                      +Vy(iX,iY)*C1*(Vy(iX,iY+1)-Vy(iX,iY-1))
        enddo
        enddo
        do iX=NXmin+1,NXmax-1
        do iY=NYmin+1,NYmax-1
            divN=-C1*(NLx(iX+1,iY)-NLx(iX-1,iY)) &
                    -C1*(NLy(iX,iY+1)-NLy(iX,iY-1))
            divU=C1*(Vx(iX+1,iY)-Vx(iX-1,iY)+Vy(iX,iY+1)-Vy(iX,iY-1))
            RHS(iX,iY)=divN+divU/dt
!    RHS(iX,iY)=divN ! non-MAC
        enddo
        enddo
! if (mod(istep,10).eq.0) then
!    iX=NXmax/4
!    iY=0
!    write(*,*) 'divU = ',C1*(Vx(iX+1,iY)-Vx(iX-1,iY)+Vy(iX,iY+1)-Vy(iX,iY-1))
! end if
        do iterate=1,itrmax  ! poisson iteration
            P2(:,:)=0.d0
            do iX=NXmin+1,NXmax-1
            do iY=NYmin+1,NYmax-1
                P2(iX,iY)=0.25d0*(P(iX-1,iY)+P(iX+1,iY)+P(iX,iY-1)+P(iX,iY+1)) &
                         -C2*0.25d0*RHS(iX,iY)
            enddo
            enddo
            delta=0.d0
            do iX=NXmin+1,NXmax-1
            do iY=NYmin+1,NYmax-1
                delta=max(delta,dabs(P2(iX,iY)-P(iX,iY)))
            enddo
            enddo
            P(:,:)=P(:,:)+alpha*(P2(:,:)-P(:,:))
            if (delta.lt.eps) goto 1
        enddo
        write(*,'(a,i8)') '**** poission solver did not converge at istep = ',istep
        stop
1       continue
! write(*,'(a,i6)') 'possison solver converged: iteration = ',iterate
    end subroutine poisson
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
    subroutine output(Vx,Vy,P)
!-----------------------------------------------------------------------------
! ■ 出力
!-----------------------------------------------------------------------------
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
        real(kind=8) C1
        character(6) :: cistep
        write(cistep,'(i6.6)') istep
        C1=1.d0/(2.d0*dX)
        open(20,file='./data/velocity.'//cistep)
! open(21,file='./data/pressure.'//cistep)
! open(22,file='./data/vorticity.'//cistep)
        do iX=NXmin,NXmax,iSep
        do iY=NYmin,NYmax,iSep
            write(20,'(4e14.6)') dX*dble(iX),dX*dble(iY),Vx(iX,iY),Vy(iX,iY)
!           write(21,'(3e14.6)') dX*dble(iX),dX*dble(iY),P(iX,iY)
!           write(22,'(3e14.6)') dX*dble(iX),dX*dble(iY), &
!           C1*((Vx(iX,iY+1)-Vx(iX,iY-1))-(Vy(iX+1,iY)-Vy(iX-1,iY)))
        enddo
! write(21,*)
! write(22,*)
        enddo
    end subroutine output
!-----------------------------------------------------------------------------
    subroutine initial_condition(Vx,Vy,P)
!-----------------------------------------------------------------------------
! ■ 初期条件 ! y成分に小さい乱数を入れる
!-----------------------------------------------------------------------------
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
        real(kind=8) X,Y
        U_tmp=Re_tmp_dble*NU/D !++!
        call random_number(Vy)
        Vy(:,:)=2.d0*Vy(:,:)-1.d0 ! [-1:1] の乱数
        Vy(:,:)=Vy(:,:)*0.001d0
        do iX=NXmin,NXmax
        do iY=NYmin,NYmax
            X=dX*real(iX,8)
            Y=dX*real(iY,8)
            R=sqrt(X**2+Y**2)
            if (R.gt.D/2.d0) then
                Vx(iX,iY)=U_tmp
            else
                Vx(iX,iY)=0.d0
            endif
        enddo
        enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Vy(NXmin,:)=0.d0
        Vy(:,NYmax)=0.d0
        Vy(:,NYmin)=0.d0
        call poisson(Vx,Vy,P)
  end subroutine initial_condition
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
    subroutine set_grid
!-----------------------------------------------------------------------------
! ■ 格子の設定
! iY1(iX) <---> iY2(iX) が円柱内 ( iX = [iXb1,iXb2] )
!-----------------------------------------------------------------------------
        real(kind=8) :: X,Y
        real(kind=8) :: twopi
        integer :: iX,iY
        dX=Xmax/dble(NXmax)
        Xmin=dble(NXmin)*dX
        Ymin=dble(NYmin)*dX
        Ymax=dble(NYmax)*dX
        iXb1=NXmax
        iXb2=NXmin
        do iX=NXmin,NXmax
            ichk=0
            do iY=NYmin,NYmax-1
                X=dX*dble(iX)
                Y=dX*dble(iY)
                if ((X**2+Y**2-(D/2.d0)**2)*(X**2+(Y+dX)**2-(D/2.d0)**2).lt.0.d0) then
                    if (ichk.eq.0) then
                        iY1(iX)=iY+1
                        ichk=1
                    else
                        iY2(iX)=iY
                    endif
                endif
            enddo
            if (ichk.eq.1) then
                iXb1=min(iXb1,iX)
                iXb2=max(iXb1,iX)
            endif
        enddo
        open(20,file='./data/grid.dat')
        do iX=iXb1,iXb2
        do iY=iY1(iX),iY2(iX)
            X=dX*dble(iX)
            Y=dX*dble(iY)
            write(20,'(2e14.6)') X,Y
        enddo
        enddo
        close(20)
        open(20,file='./data/circle.dat')
        twopi=8.d0*atan(1.d0)
        do i=0,200
            write(20,'(2e14.6)') (D/2.d0)*cos(real(i,8)*twopi/200), &
                                  (D/2.d0)*sin(real(i,8)*twopi/200)
        end do
        close(20)
    end subroutine set_grid
    subroutine  Cal_abs_vector(d_Vx,d_Vy,abs_dV)
        real(kind=8) :: d_Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: d_Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: abs_dV
        real(8) X,Y
        abs_dV=0.d0
        do iX=NXmin,NXmax
        do iY=NYmin,NYmax
            abs_dV = abs_dV + d_Vx(iX,iY)**2 +d_Vy(iX,iY)**2
        enddo
        enddo
        abs_dV = abs_dV**(0.5d0)
    end subroutine Cal_abs_vector
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine create_r_matrix(Vx,r_tr,time)
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: r_tr(:,:)
        integer time
        integer i,j,k
        do i= 1,tate_y
            do j = 1,yoko_x
                r_tr(time,(i-1)*yoko_x+j)=Vx(j+heikouidou_x,i+heikouidou_y)
            enddo
        enddo
    end subroutine create_r_matrix
    subroutine create_Wout_matrix(r_tr,s_tr,w_out)
        integer i,j,k
        real(kind=8) r_tr(:,:)
        real(kind=8) s_tr(:,:)
        real(kind=8) w_out(:,:)
        real(kind=8) tmp_1(RC_NODE,RC_NODE)
	    real(kind=8) tmp_2(RC_NODE,OUT_NODE)
	    real(kind=8) tmp_3(RC_NODE,RC_NODE)
	    real(kind=8) e(RC_NODE,RC_NODE)
	 	real(kind=8) inverse(RC_NODE,RC_NODE)
	 	real(kind=8) :: beta=1.d-3
	!_________________________________
	!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
	    real(kind=8) work(64*RC_NODE)
		integer :: lwork = 64*RC_NODE
		integer ipiv(1:RC_NODE)
		integer info
	!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    !---------------------------------
        	e=0.d0
        	tmp_1 =0.d0
        	tmp_2 =0.d0
        	tmp_3 =0.d0
        	w_out =0.d0
    		do i=1,RC_NODE
    			e(i,i)=1.d0
    		enddo
        	do i=1,RC_NODE
        		do j=1,RC_NODE
        			do k=1,TRANING_STEP
        				tmp_1(i,j) = tmp_1(i,j) + r_tr(k,i)*r_tr(k,j)
        			enddo
        		enddo
        	enddo
    		tmp_1(1:RC_NODE,1:RC_NODE)=tmp_1(1:RC_NODE,1:RC_NODE)+beta*e(1:RC_NODE,1:RC_NODE)

!tmp_2はRC_NODE行OUT_NODE列の行列
            do j=1,OUT_NODE
	    	do i=1,RC_NODE
	    		do k=1,TRANING_STEP
	    			tmp_2(i,j) = tmp_2(i,j) + r_tr(k,i)*s_tr(k,j)
	    		enddo
	    	enddo
	    	enddo
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!逆行列作成++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        	tmp_3=tmp_1
    		call dgetrf(RC_NODE, RC_NODE, tmp_1, RC_NODE, ipiv, info)
    		call dgetri(RC_NODE,tmp_1, RC_NODE, ipiv, work, lwork, info)
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!逆行列確かめ++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    	    inverse=0.d0
    	    do i=1,RC_NODE
    	    	do j=1,RC_NODE
    	    		do k=1,RC_NODE
    	    			inverse(i,j)=inverse(i,j) + tmp_1(i,k)*tmp_3(k,j)
    	    		enddo
    	    		if(abs(inverse(i,j))<1.d-6 .and. i/=j) inverse(i,j)=0.d0
    	    	enddo
    	    enddo
    	    do i=1,RC_NODE
    	    	if(mod(i,100) == 0.d0) write(*,*) inverse(i,100)
    	    enddo
    	    do i=1,RC_NODE
    	    	do j=1,RC_NODE
    	    		if(abs(inverse(i,j))>1.d-6 .and. i/=j) then
    	    			write(*,*) i,j
    	    			write(*,*) inverse(i,j)
    	    		endif
    	    	enddo
    	    enddo
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!W_out作成（OUT_NODE行RC_NODE列）++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            do j=1,OUT_NODE
	    	do i=1,RC_NODE
	    		do k=1,RC_NODE
	    			w_out(i,j) = w_out(i,j) + tmp_1(i,k)* tmp_2(k,j)
	    		enddo
	    	enddo
	    	enddo
    end subroutine create_Wout_matrix
!---------------------------------,--------------------------------------------
    subroutine RC_OWN(Vx,S_rc,W_out)
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: S_rc(:,:)
        real(kind=8) :: W_out(:,:)
        real(kind=8) :: r_tmp(1:RC_NODE)
        integer i,j,k
        do i= 1,tate_y
            do j = 1,yoko_x
                r_tmp((i-1)*yoko_x+j) = Vx(j+heikouidou_x, i+ heikouidou_y)
            enddo
        enddo
        S_rc(istep,:) = 0.d0
        do j=1,OUT_NODE
        do i=1,RC_NODE
            S_rc(istep,j) = S_rc(istep,j) + r_tmp(i)*W_out(i,j)
        enddo
        enddo
    end subroutine RC_OWN

!-S_rc,W_out)----------------------------------------------------------------------------
end module cylinder
!-----------------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine rc_traning_own_karman(in_node00,out_node00,rc_node00,traning_step00,rc_step00,&
                    u_trT,s_trT,u_rcT,s_rcT,w_outT)
  use cylinder
    implicit none
    integer(4), intent(inout) :: in_node00,out_node00,rc_node00,traning_step00,rc_step00
    real(8),    intent(inout) :: w_outT(out_node00,rc_node00)
    real(8),    intent(inout) ::u_trT(in_node00,traning_step00) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_trT(out_node00,traning_step00)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) ::u_rcT(in_node00 ,rc_step00) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_rcT(out_node00,rc_step00)  !出力次元数、列サイズはトレーニング時間
    real(kind=8) :: R_tr(traning_step00,rc_node00)
    real(kind=8) :: W_out(rc_node00,out_node00)
    real(kind=8) :: U_tr (traning_step00,in_node00)
    real(kind=8) :: S_tr (traning_step00,out_node00)
    real(kind=8) :: U_rc (rc_step00,in_node00)
    real(kind=8) :: S_rc (rc_step00,out_node00)
    real(kind=8) :: S_rc_data(rc_step00,out_node00)
    real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: DATA(2,1:3),RCerr
    integer iX,iY,i,j
    integer Re_tmp_int,iERR


!=================================================================================================================
!-----------------------------------------------------------------------------------------------------------------
!初期化
!    open(82,file='./data_lyap_end/lyapnov_end.dat',status='replace')
!    close(82)
!    open(46,file='RC_err.dat',status='replace')
!    close(46)
!    OUT_NODE = out_node00
!    IN_NODE = in_node00
!    TRANING_STEP =traning_step00
!    RC_STEP = rc_step
!    call allocate_mat(in_node00,out_node00,rc_node00,traning_step00,rc_step00)

    OUT_NODE = out_node00
    IN_NODE = in_node00
    TRANING_STEP = traning_step00
    RC_STEP = rc_step00
    
    do i=1,IN_NODE
    do j=1,TRANING_STEP
        U_tr(j,i) = u_trT(i,j)
    enddo
    enddo
    do i=1,OUT_NODE
    do j=1,TRANING_STEP
        S_tr(j,i) = s_trT(i,j)
    enddo
    enddo
!    do i=1,IN_NODE
!    do j=1,RC_STEP
!        U_rc(j,i) = u_rcT(i,j)
!    enddo
!    enddo
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "    welcome to  Fortran90 !    "
    write(*,*) "-------------------------------"
    WRITE(*,*) "IN_NODE      ",IN_NODE
    WRITE(*,*) "OUT_NODE     ",OUT_NODE
    WRITE(*,*) "RC_NODE      ",RC_NODE
    WRITE(*,*) "TRANING_STEP ",TRANING_STEP
    WRITE(*,*) "RC_STEP      ",RC_STEP
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""
!=================================================================================================================
    call set_grid
    
!【Reのdoループ 】
    do Re_tmp_int = 10,10,2
!【TYU,TYOのdoループ 】
    do iERR = 10,10
!    TYU=dble(iERR)*0.1d0
    TYO=dble(iERR)*1.d0
!    Re_tmp_int =30
    
        Re_tmp_dble =dble(Re_tmp_int)
        RCerr=0.d0

        call initial_condition(Vx,Vy,P)
!  call output_V_P(Vx,Vy,P,0)
        write(Re_tmp_name,'(i3.3)') Re_tmp_int
        write(*,*) ""
        write(*,*) "============================================= "
        write(*,*) "START >>  Re ==" ,Re_tmp_dble
        write(*,*) "START >>  TYO ==" ,TYO
        write(*,*) "============================================= "



!流れが定常or周期状態になるまでスキップ
        write(*,*) "=========================================="
        write(*,*) "     SKIP STEP"
        write(*,*) "=========================================="
        do istep=1,skip_step
!=================================================================================================================
!-----------------------------------------------------------------------------------------------------------------
!ディスプレイ表示(一つ前のステップの値が出力istep=10ならistep=9の値)
!-----------------------------------------------------------------------------------------------------------------
            if (mod(istep,iout_display).eq.0) write(*,*) 'Skip_step = ',istep
            if (mod(istep,iout_display).eq.0) write(*,*) '    Re    = ',Re_tmp_int
            if (mod(istep,iout_display).eq.0) write(*,*) '    U     = ',U_tmp
            if (mod(istep,iout_display).eq.0) write(*,*) "   TYO    = " ,TYO
            if (mod(istep,iout_display).eq.0) write(*,200) '-------------    Vx(NXmax/2,0)=',Vx(NXmax/2,0)
            call march(Vx,Vy,P,U_tr,0)
        enddo
  
!        Vx_tmp(:,:) = Vx(:,:)
!        Vy_tmp(:,:) = Vy(:,:)


        write(*,*) "=========================================="
        write(*,*) "     TRANING STEP"
        write(*,*) "=========================================="
        do istep=1,TRANING_STEP
            if (mod(istep,iout_display).eq.0) write(*,*) 'TRANING_step = ',istep
            if (mod(istep,iout_display).eq.0) write(*,*) '    Re    = ',Re_tmp_int
            if (mod(istep,iout_display).eq.0) write(*,*) '    U     = ',U_tmp
            if (mod(istep,iout_display).eq.0) write(*,*) "   TYO    = " ,TYO
            if (mod(istep,iout_display).eq.0) write(*,200) '-------------    Vx(NXmax/2,0)=',Vx(NXmax/2,0)
            call march(Vx,Vy,P,U_tr,1)
            if (mod(istep,  iout).eq.0) call output(Vx,Vy,P)
            call create_r_matrix(Vx,R_tr,istep)
        enddo
        write(*,*) "=========================================="
        write(*,*) "     INVERSE MATRIX CALCULATION"
        write(*,*) "=========================================="
        call create_Wout_matrix(R_tr,S_tr,W_out)
        close(40)
!=================================================================================================================%%
        Vx_tmp(:,:) = Vx(:,:)
        Vy_tmp(:,:) = Vy(:,:)
        
        U_rc(1,in_node) =S_tr(traning_step,out_node)
        write(*,*) "=========================================="
        write(*,*) "     RC STEP"
        write(*,*) "=========================================="
        do istep=1,RC_STEP
            call march(Vx,Vy,P,U_rc,2)
            call RC_OWN(Vx,S_rc,W_out)
            if(istep<RC_STEP)  U_rc(istep+1,1)=S_rc(istep,1)
        
            Vx_tmp(:,:) = Vx(:,:)
            Vy_tmp(:,:) = Vy(:,:)
        enddo
    enddo
    enddo
    s_rc(:,:)=S_rc(:,:)/s_rc(1,1)


    do i=1,out_node
    do j=1,rc_step
        s_rcT(i,j) = s_rc(j,i)
    enddo
    enddo
    do i=1,out_node
    do j=1,rc_node
        w_outT(i,j) = W_out(j,i)
    enddo
    enddo

100 format(a,i6,a,f15.10)
200 format(a,f15.10)
end subroutine rc_traning_own_karman
!-----------------------------------------------------------------------------
